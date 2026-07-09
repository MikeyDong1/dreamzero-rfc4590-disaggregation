#!/usr/bin/env python3
"""SFTP push/pull to gnr17408 (sdp@10.54.109.211). Key/agent auth only.

Usage:
    python scp_gnr17408.py put <local> <remote>
    python scp_gnr17408.py get <remote> <local>

Prints periodic progress (every ~256 MB) so large transfers are observable.
Skips re-upload if the remote file already matches the local size.
"""
import os
import sys
import paramiko

HOST, USER = "10.54.109.211", "sdp"


def _connect():
    c = paramiko.SSHClient()
    c.load_system_host_keys()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, allow_agent=True, look_for_keys=True,
              timeout=30, banner_timeout=30, auth_timeout=30)
    return c


_last = [0]


def _progress(done, total):
    # print every 256 MB and at the end
    if done - _last[0] >= 256 * 1024 * 1024 or done == total:
        pct = (100.0 * done / total) if total else 0.0
        print(f"  {done/1e9:.2f}/{total/1e9:.2f} GB ({pct:.1f}%)", flush=True)
        _last[0] = done


def main():
    mode, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    c = _connect()
    sftp = c.open_sftp()
    if mode == "put":
        local_size = os.path.getsize(src)
        try:
            rstat = sftp.stat(dst)
            if rstat.st_size == local_size:
                print(f"SKIP (already present, {local_size/1e9:.2f} GB): {dst}", flush=True)
                sftp.close(); c.close(); return
        except IOError:
            pass
        print(f"PUT {src} -> {USER}@{HOST}:{dst} ({local_size/1e9:.2f} GB)", flush=True)
        sftp.put(src, dst, callback=_progress)
        # verify size
        rstat = sftp.stat(dst)
        ok = rstat.st_size == local_size
        print(f"PUT {'OK' if ok else 'SIZE-MISMATCH'}: remote={rstat.st_size} local={local_size}", flush=True)
    else:
        print(f"GET {USER}@{HOST}:{src} -> {dst}", flush=True)
        sftp.get(src, dst, callback=_progress)
        print("GET OK", flush=True)
    sftp.close(); c.close()


if __name__ == "__main__":
    main()
