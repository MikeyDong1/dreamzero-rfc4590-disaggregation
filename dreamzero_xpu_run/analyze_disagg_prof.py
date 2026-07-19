#!/usr/bin/env python3
"""Offline analysis of the disaggregated DreamZero per-process phase events.

Reads $DZ_PROF_DIR/events.<pid>.jsonl (one file per process: 1 encode, 4 denoise
TP ranks, 1 decode) plus result_multi.json (orchestrator-side req wall marks),
and reconstructs the cross-process per-request timeline.

Because every event carries an ABSOLUTE time.time() start/end on the single host
clock, we can align a request's encode-end against its denoise-start (on a
different process/device) and read the GAP directly:

  request wall  = generate() end - start                          (orchestrator)
  encode        = encode phase dur on the encode pid
  gap E->D      = diffuse.t_start - encode.t_end                  (transport enc->den + queue)
  denoise       = diffuse phase dur on a representative denoise rank
  gap D->De     = postprocess.t_start - diffuse.t_end             (transport den->dec + queue)
  decode        = postprocess phase dur on the decode pid
  unaccounted   = request wall - (encode + gapED + denoise + gapDDe + decode)

Matching: within each process, phase calls are serial; the Nth encode call is
request N (seq 0 = warmup). Denoise has 4 rank pids each with its own seq; we
pick the pid whose role=='denoise'/'DENOISE' with the most events as the
representative (ranks are symmetric lock-step).
"""
import os, sys, json, glob
from collections import defaultdict


def load_events(prof_dir):
    ev_by_pid = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(prof_dir, 'events.*.jsonl'))):
        pid = os.path.basename(path).split('.')[1]
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                ev_by_pid[pid].append(rec)
    return ev_by_pid


def role_of_pid(events):
    """Infer a pid's stage role from the phase methods it emitted."""
    methods = {e['method'] for e in events}
    if 'diffuse' in methods or '_run_dit_loop' in methods:
        return 'denoise'
    if 'encode' in methods or '_encode_image' in methods:
        return 'encode'
    if 'postprocess' in methods or 'decode_video_latents' in methods:
        return 'decode'
    # transport-only or ambiguous
    roles = {e.get('role') for e in events}
    return next((r for r in roles if r and r != '?'), 'unknown')


def phase_seq(events, method):
    """Return list of records for `method` ordered by seq."""
    recs = [e for e in events if e['method'] == method]
    recs.sort(key=lambda e: e.get('seq', 0))
    return recs


def main():
    prof_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('DZ_PROF_DIR', '.')
    result_json = sys.argv[2] if len(sys.argv) > 2 else os.path.join(prof_dir, '..', 'metrics', 'result_multi.json')

    ev_by_pid = load_events(prof_dir)
    if not ev_by_pid:
        print(json.dumps({'error': 'no event files found in %s' % prof_dir}))
        return

    pid_roles = {pid: role_of_pid(evs) for pid, evs in ev_by_pid.items()}
    # group pids by role
    by_role = defaultdict(list)
    for pid, role in pid_roles.items():
        by_role[role].append(pid)

    # representative denoise rank = pid with most diffuse events
    def pick_rep(role, method):
        cands = by_role.get(role, [])
        if not cands:
            return None
        return max(cands, key=lambda p: len(phase_seq(ev_by_pid[p], method)))

    enc_pid = pick_rep('encode', 'encode')
    den_pid = pick_rep('denoise', 'diffuse')
    dec_pid = pick_rep('decode', 'postprocess')

    enc = phase_seq(ev_by_pid[enc_pid], 'encode') if enc_pid else []
    den = phase_seq(ev_by_pid[den_pid], 'diffuse') if den_pid else []
    dec = phase_seq(ev_by_pid[dec_pid], 'postprocess') if dec_pid else []

    # orchestrator req marks
    req_marks = []
    try:
        with open(os.path.abspath(result_json)) as f:
            rj = json.load(f)
        req_marks = rj.get('req_wall_marks', [])
        server_perf = rj.get('server_denoise_perf')
    except Exception:
        server_perf = None

    n = max(len(enc), len(den), len(dec))
    rows = []
    for i in range(n):
        e = enc[i] if i < len(enc) else None
        d = den[i] if i < len(den) else None
        p = dec[i] if i < len(dec) else None
        rm = next((m for m in req_marks if m.get('req') == i), None)

        gap_ed = (d['t_start'] - e['t_end']) if (e and d) else None
        gap_dde = (p['t_start'] - d['t_end']) if (d and p) else None
        wall = rm['dur_s'] if rm else None
        enc_s = e['dur_s'] if e else None
        den_s = d['dur_s'] if d else None
        dec_s = p['dur_s'] if p else None

        accounted = sum(x for x in [enc_s, gap_ed, den_s, gap_dde, dec_s] if x is not None)
        unacc = (wall - accounted) if wall is not None else None

        rows.append({
            'req': i,
            'is_warmup': i == 0,
            'wall_s': wall,
            'encode_s': enc_s,
            'gap_enc2den_s': gap_ed,
            'denoise_s': den_s,
            'gap_den2dec_s': gap_dde,
            'decode_s': dec_s,
            'sum_accounted_s': accounted,
            'unaccounted_s': unacc,
            'carrier_csf_in': d.get('carrier_csf_in') if d else None,
            'reset_reason': d.get('reset_reason') if d else None,
            'denoise_state_csf_after': d.get('denoise_state_csf_after') if d else None,
        })

    # sub-phase detail per stage (steady-state = requests 2..N, skip warmup+first-warm)
    def subphase_summary(pid, methods):
        out = {}
        if not pid:
            return out
        for m in methods:
            recs = phase_seq(ev_by_pid[pid], m)
            durs = [r['dur_s'] for r in recs]
            # steady-state: drop seq 0 (warmup) and seq 1 (first warm) if present
            steady = [r['dur_s'] for r in recs if r.get('seq', 0) >= 2]
            out[m] = {
                'n_calls': len(recs),
                'all_durs_s': [round(x, 4) for x in durs],
                'steady_mean_s': (round(sum(steady) / len(steady), 4) if steady else None),
            }
        return out

    enc_sub = subphase_summary(enc_pid, ['encode', '_encode_text', '_encode_image', '_encode_observation_latents', 'pack_stage_state'])
    den_sub = subphase_summary(den_pid, ['diffuse', '_kv_populate_cross', '_prefill_kv_cache', '_run_dit_loop', 'unpack_stage_state', 'pack_stage_state'])
    dec_sub = subphase_summary(dec_pid, ['postprocess', 'decode_video_latents', '_denormalize_action', 'unpack_stage_state'])

    # transport sanitize bytes per producer pid
    transport = {}
    for pid, evs in ev_by_pid.items():
        san = [e for e in evs if e['method'] == 'sanitize_transport_tensor']
        if san:
            transport[pid] = {
                'role': pid_roles[pid],
                'n_tensor_copies': len(san),
                'total_mib': round(sum(e.get('nbytes', 0) for e in san) / 2**20, 3),
                'total_d2h_s': round(sum(e['dur_s'] for e in san), 4),
            }

    # steady-state aggregate (requests >=2, excluding warmup/first-warm)
    steady_rows = [r for r in rows if r['req'] >= 2 and r['wall_s'] is not None]

    def _mean(key):
        vals = [r[key] for r in steady_rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    steady_agg = {
        'n': len(steady_rows),
        'wall_s': _mean('wall_s'),
        'encode_s': _mean('encode_s'),
        'gap_enc2den_s': _mean('gap_enc2den_s'),
        'denoise_s': _mean('denoise_s'),
        'gap_den2dec_s': _mean('gap_den2dec_s'),
        'decode_s': _mean('decode_s'),
        'unaccounted_s': _mean('unaccounted_s'),
        'denoise_dit_loop_s': (den_sub.get('_run_dit_loop', {}) or {}).get('steady_mean_s'),
        'denoise_prefill_s': (den_sub.get('_prefill_kv_cache', {}) or {}).get('steady_mean_s'),
        'denoise_unpack_s': (den_sub.get('unpack_stage_state', {}) or {}).get('steady_mean_s'),
    }

    report = {
        'prof_dir': prof_dir,
        'pids': pid_roles,
        'representative_pids': {'encode': enc_pid, 'denoise': den_pid, 'decode': dec_pid},
        'per_request': rows,
        'steady_state_mean': steady_agg,
        'encode_subphases': enc_sub,
        'denoise_subphases': den_sub,
        'decode_subphases': dec_sub,
        'transport_d2h': transport,
        'server_denoise_perf_rpc': server_perf,
    }
    out_path = os.path.join(prof_dir, 'timeline_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)

    # human-readable
    print('=== DreamZero disaggregated per-request timeline (s) ===')
    print('gapED = enc.end->den.start (transport+queue); gapDDe = den.end->dec.start; csf = carrier current_start_frame')
    hdr = ('req', 'wall', 'encode', 'gapED', 'denoise', 'gapDDe', 'decode', 'unacc', 'csfIn')
    print('%-4s %7s %7s %7s %8s %7s %7s %7s %6s' % hdr)
    for r in rows:
        def f(x, w=7, p=3):
            return ('%*.*f' % (w, p, x)) if isinstance(x, (int, float)) else ('%*s' % (w, '-'))
        print('%-4s %7s %7s %7s %8s %7s %7s %7s %6s' % (
            str(r['req']) + ('*' if r['is_warmup'] else ''),
            f(r['wall_s']), f(r['encode_s']), f(r['gap_enc2den_s']),
            f(r['denoise_s'], 8), f(r['gap_den2dec_s']), f(r['decode_s']),
            f(r['unaccounted_s']),
            f(r['carrier_csf_in'], 6, 0)))
    print()
    print('=== steady-state mean (req>=2) ===')
    print(json.dumps(steady_agg, indent=2))
    print()
    print('[ANALYSIS_JSON]=%s' % out_path)


if __name__ == '__main__':
    main()
