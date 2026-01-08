#!/usr/bin/env python3
"""
webrtc_probe.py

Network-level probe for Isaac Sim / OmniGibson WebRTC streaming.

- TCP connect test (8211/49100)
- UDP reachability probe (49100 is often UDP for WebRTC)
- Optional HTTP HEAD test for 8211 endpoint

Usage examples:
  python webrtc_probe.py --host 192.168.1.128
  python webrtc_probe.py --host 192.168.1.128 --http-port 8211 --webrtc-port 49100
  python webrtc_probe.py --host 192.168.1.128 --skip-http
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


def tcp_connect(host: str, port: int, timeout: float) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, "TCP connect succeeded"
    except OSError as e:
        return False, f"TCP connect failed: {e}"


def udp_probe(host: str, port: int, timeout: float) -> tuple[bool, str]:
    """
    UDP is connectionless; there is no true "connect".
    Strategy:
      - connect() a UDP socket (sets default remote)
      - send a probe payload
      - try recv() briefly:
          - if we get ConnectionRefused -> ICMP Port Unreachable (port closed / rejected)
          - if timeout -> inconclusive but often means "no immediate ICMP", which is OK for UDP reachability
          - if data received -> port responded (rare for WebRTC but strong signal)
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        s.send(b"webrtc-udp-probe")
        try:
            data = s.recv(2048)
            return True, f"UDP probe got response ({len(data)} bytes) -> reachable and responding"
        except socket.timeout:
            # Most UDP services won't respond to random packets; timeout here can still be OK.
            return True, "UDP probe sent; no ICMP error observed (timeout) -> UDP path likely reachable (inconclusive but OK)"
        except ConnectionRefusedError:
            # Linux surfaces ICMP Port Unreachable as ConnectionRefused on recv after send
            return False, "UDP probe resulted in ICMP Port Unreachable -> port closed or firewall REJECT"
    except OSError as e:
        return False, f"UDP probe failed: {e}"
    finally:
        s.close()


def http_head(url: str, timeout: float) -> tuple[bool, str]:
    req = Request(url, method="HEAD")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return True, f"HTTP HEAD {resp.status} {resp.reason}"
    except HTTPError as e:
        # HTTPError is also a valid HTTP response (e.g., 301/302/404)
        return True, f"HTTP HEAD {e.code} {e.reason}"
    except URLError as e:
        return False, f"HTTP request failed: {e}"


def print_result(name: str, ok: bool, msg: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {msg}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Isaac Sim / OmniGibson WebRTC ports.")
    parser.add_argument("--host", required=True, help="Streaming server IP/hostname (e.g., 192.168.1.128)")
    parser.add_argument("--http-port", type=int, default=8211, help="HTTP port for webrtc-client page (default: 8211)")
    parser.add_argument("--webrtc-port", type=int, default=49100, help="WebRTC port (default: 49100)")
    parser.add_argument("--timeout", type=float, default=2.0, help="Timeout seconds for each probe (default: 2.0)")
    parser.add_argument("--skip-http", action="store_true", help="Skip HTTP probe (useful if 8211 is not enabled)")
    parser.add_argument("--path", default="/streaming/webrtc-client", help="HTTP path to probe (default: /streaming/webrtc-client)")
    args = parser.parse_args()

    host = args.host
    timeout = args.timeout

    print(f"== WebRTC network probe ==")
    print(f"Host: {host}")
    print(f"HTTP:  {args.http_port}{args.path}")
    print(f"WebRTC: {args.webrtc_port} (TCP+UDP probes)")
    print()

    # TCP probes
    ok, msg = tcp_connect(host, args.webrtc_port, timeout)
    print_result(f"TCP {args.webrtc_port}", ok, msg)

    ok_udp, msg_udp = udp_probe(host, args.webrtc_port, timeout)
    print_result(f"UDP {args.webrtc_port}", ok_udp, msg_udp)

    if not args.skip_http:
        ok_http_tcp, msg_http_tcp = tcp_connect(host, args.http_port, timeout)
        print_result(f"TCP {args.http_port}", ok_http_tcp, msg_http_tcp)

        url = f"http://{host}:{args.http_port}{args.path}"
        ok_http, msg_http = http_head(url, timeout)
        print_result(f"HTTP HEAD {url}", ok_http, msg_http)

    print("\n== Interpretation hints ==")
    print("- If TCP 49100 passes AND UDP 49100 does NOT show ICMP unreachable, WebRTC port is likely reachable.")
    print("- If TCP 8211 fails/refused, the web client HTTP server is not enabled or not listening externally.")
    print("- 'UDP timeout' is often OK: many UDP services don't reply to random probes.")
    print("- If UDP shows 'ICMP Port Unreachable', the UDP port is closed or firewall is rejecting it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
