# SPDX-License-Identifier: Apache-2.0
"""Worker RPC extension: report per-rank param counts/shapes to check TP replication.

Combine with the existing DreamZeroVideoExportWorkerExtension via vLLM-Omni's
dynamic worker_extension_cls composition (Omni(worker_extension_cls=...) accepts
one class; we compose both onto one class here since WorkerWrapperBase requires
a single worker_extension_cls string).

IMPORTANT: collective_rpc(exec_all_ranks=True) executes the method on every
worker process, but the multiproc executor's result channel only has a
result_mq on rank 0 -- so the Python return value that reaches the caller is
ONLY rank 0's. To actually see every rank's report, each rank writes its own
JSON file to a shared (bind-mounted) directory; the caller RPC still runs on
all ranks (so all files get written) and then the driver script reads the
directory directly instead of relying on the RPC return value.
"""
from __future__ import annotations

import json
import os

import torch

from vllm_omni.diffusion.models.dreamzero.video_export_worker import (
    DreamZeroVideoExportWorkerExtension,
)

#: Shared directory (bind-mounted into every worker process) where each rank
#: drops its own report file. Overridable via env for portability.
_REPORT_DIR = os.environ.get("TP_PROBE_REPORT_DIR", "/workspace/probe_reports")


def _module_report(mod: torch.nn.Module | None) -> dict:
    if mod is None:
        return {"present": False}
    total_params = sum(p.numel() for p in mod.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in mod.parameters())
    # A few representative leaf-parameter shapes (sorted names) to compare shapes
    # (not just counts) across ranks -- sharding usually halves/quarters one dim.
    named = sorted(mod.named_parameters(), key=lambda kv: kv[0])
    sample = [
        {"name": n, "shape": list(p.shape), "dtype": str(p.dtype)}
        for n, p in named[:3]
    ]
    return {
        "present": True,
        "total_params": int(total_params),
        "total_bytes": int(total_bytes),
        "total_gib": total_bytes / (1024**3),
        "sample_param_shapes": sample,
    }


class TPReplicationProbeExtension(DreamZeroVideoExportWorkerExtension):
    """Adds tp_replication_report() / dump_gpu_mem_stats() alongside the export RPCs.

    Both write one JSON file per rank into _REPORT_DIR instead of relying on the
    RPC return value (which only surfaces rank 0's result — see module docstring).
    """

    def _rank_world(self) -> tuple[int, int]:
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        return int(rank), int(world_size)

    def tp_replication_report(self) -> dict:
        runner = self.model_runner
        if runner is None or runner.pipeline is None:
            raise RuntimeError("DreamZero pipeline is not initialized on this worker.")
        pipe = runner.pipeline
        dev = torch.accelerator.current_device_index()
        rank, world_size = self._rank_world()

        report = {
            "rank": rank,
            "world_size": world_size,
            "device_index": int(dev),
            "text_encoder": _module_report(getattr(pipe, "text_encoder", None)),
            "image_encoder": _module_report(getattr(pipe, "image_encoder", None)),
            "vae": _module_report(getattr(pipe, "vae", None)),
            "transformer": _module_report(getattr(pipe, "transformer", None)),
        }

        # A couple of representative DiT sub-layer shapes to prove sharding
        # (e.g. qkv/ffn out_features reduced by world_size on TP-sharded layers).
        transformer = getattr(pipe, "transformer", None)
        if transformer is not None and hasattr(transformer, "blocks") and len(transformer.blocks) > 0:
            block0 = transformer.blocks[0]
            block_detail = {}
            for attr_path in ("self_attn.qkv", "self_attn.o", "ffn.0", "ffn.2"):
                obj = block0
                ok = True
                for part in attr_path.split("."):
                    obj = getattr(obj, part, None)
                    if obj is None:
                        ok = False
                        break
                if ok and hasattr(obj, "weight"):
                    block_detail[attr_path] = list(obj.weight.shape)
            report["transformer_block0_layer_shapes"] = block_detail

        os.makedirs(_REPORT_DIR, exist_ok=True)
        with open(os.path.join(_REPORT_DIR, f"tp_report_rank{rank}.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)
        return report

    def dump_gpu_mem_stats(self) -> dict:
        """Same content as gpu_mem_stats(), but written per-rank to _REPORT_DIR."""
        stats = self.gpu_mem_stats()
        rank, world_size = self._rank_world()
        stats["rank"] = rank
        stats["world_size"] = world_size
        os.makedirs(_REPORT_DIR, exist_ok=True)
        with open(os.path.join(_REPORT_DIR, f"mem_rank{rank}.json"), "w") as f:
            json.dump(stats, f, indent=2, default=str)
        return stats
