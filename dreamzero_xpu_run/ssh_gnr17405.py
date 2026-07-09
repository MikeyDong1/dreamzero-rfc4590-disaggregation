#!/usr/bin/env python3
"""Run a remote command on gnr17405 via paramiko (password auth).

Usage:
    python ssh_gnr17405.py "command to run"
    python ssh_gnr17405.py --timeout 600 "long command"
"""
import sys
import argparse
import paramiko

HOST = "10.54.109.207"
USER = "sdp"
PASSWORD = "sdpIntel"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("cmd", nargs="+")
    args = ap.parse_args()
    command = " ".join(args.cmd)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=30,
                   banner_timeout=30, auth_timeout=30)

    # Use a login shell so env (conda, modules) is loaded.
    stdin, stdout, stderr = client.exec_command(command, timeout=args.timeout,
                                                get_pty=False)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    # Force UTF-8 to survive progress bars / non-cp1252 chars on Windows console.
    enc = (sys.stdout.encoding or "utf-8")
    out_b = out.encode(enc, "replace")
    sys.stdout.buffer.write(out_b)
    if err:
        sys.stderr.buffer.write(("\n[stderr]\n" + err).encode(enc, "replace"))
    sys.stdout.buffer.write(f"\n[exit {rc}]\n".encode(enc, "replace"))
    client.close()
    sys.exit(rc)


if __name__ == "__main__":
    main()
