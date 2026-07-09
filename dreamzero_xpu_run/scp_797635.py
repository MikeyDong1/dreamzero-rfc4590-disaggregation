#!/usr/bin/env python3
"""SFTP push/pull to srf797635. Usage:
    python scp_797635.py put <local> <remote>
    python scp_797635.py get <remote> <local>
"""
import sys, paramiko
HOST, USER, PASSWORD = "10.23.14.76", "sdp", "sdp_intel"

def main():
    mode, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    t = paramiko.Transport((HOST, 22)); t.connect(username=USER, password=PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(t)
    if mode == "put":
        sftp.put(src, dst)
    else:
        sftp.get(src, dst)
    sftp.close(); t.close()
    print(f"{mode} OK: {src} -> {dst}")

if __name__ == "__main__":
    main()
