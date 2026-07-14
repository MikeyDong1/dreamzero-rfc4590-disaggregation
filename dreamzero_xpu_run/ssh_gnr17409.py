#!/usr/bin/env python3
"""Run a remote command on gnr17409 (sdp@10.54.109.214) via paramiko.

Key/agent auth only (no password embedded, per ssh-node-connect skill rules).

Usage:
    python ssh_gnr17409.py "command to run"
    python ssh_gnr17409.py --timeout 600 "long command"
"""
import sys
import argparse
import paramiko

HOST = "10.54.109.214"
USER = "sdp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("cmd", nargs="+")
    args = ap.parse_args()
    command = " ".join(args.cmd)

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, allow_agent=True, look_for_keys=True,
                   timeout=30, banner_timeout=30, auth_timeout=30)

    stdin, stdout, stderr = client.exec_command(command, timeout=args.timeout,
                                                get_pty=False)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    enc = (sys.stdout.encoding or "utf-8")
    sys.stdout.buffer.write(out.encode(enc, "replace"))
    if err:
        sys.stderr.buffer.write(("\n[stderr]\n" + err).encode(enc, "replace"))
    sys.stdout.buffer.write(f"\n[exit {rc}]\n".encode(enc, "replace"))
    client.close()
    sys.exit(rc)


if __name__ == "__main__":
    main()
