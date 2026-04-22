"""Sidecar HTTP server that serves overlay videos with proper Content-Type + CORS + Range.

Streamlit's built-in static file handler forces text/plain on any extension outside
a small whitelist (images, fonts, pdf, json, xml) -- MP4 is not allowed, so browsers
won't play the overlay videos. This tiny server fills that gap.

Supports HTTP Range requests so mobile browsers (which use partial-content fetches for
<video> streaming) can seek and buffer properly.

Runs on port 8502 by default, serves `FollowThrough/demo/static/`.

Use:
    python3 serve_videos.py [--host 0.0.0.0] [--port 8502]
"""

from __future__ import annotations

import argparse
import http.server
import mimetypes
import os
import re
import socketserver
from pathlib import Path

DEFAULT_PORT = 8502
SERVE_DIR = Path(__file__).resolve().parent / "static"


class CORSRangeHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler + CORS + Range support."""

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        # Honour HTTP Range requests -- mobile browsers need this for <video>.
        range_hdr = self.headers.get("Range")
        if not range_hdr:
            super().do_GET()
            return

        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404)
            return

        m = re.match(r"bytes=(\d+)-(\d*)", range_hdr)
        if not m:
            self.send_error(416)
            return

        file_size = os.path.getsize(path)
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else file_size - 1
        end = min(end, file_size - 1)
        if start > end:
            self.send_error(416)
            return

        length = end - start + 1
        ctype = self.guess_type(path)

        self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()

        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0", help="Bind address (0.0.0.0 = all interfaces)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()

    mimetypes.add_type("video/mp4", ".mp4")
    mimetypes.add_type("video/webm", ".webm")

    os.chdir(SERVE_DIR)

    with ThreadedTCPServer((args.host, args.port), CORSRangeHandler) as httpd:
        print(f"Serving {SERVE_DIR} on http://{args.host}:{args.port}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
