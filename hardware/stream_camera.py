"""Broadcast the robot camera as an MJPEG stream viewable in any browser.

Usage (on the robot):
    python hardware/stream_camera.py

Then open  http://<robot-ip>:8000  on your laptop.
"""
from __future__ import annotations

import argparse
import io
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--port",   type=int, default=8000)
parser.add_argument("--width",  type=int, default=640)
parser.add_argument("--height", type=int, default=480)
args = parser.parse_args()

from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput


_PAGE = b"""<!DOCTYPE html>
<html><head><title>Robot Camera</title></head>
<body style="margin:0;background:#000">
<img src="/stream.mjpg" style="max-width:100%;display:block;margin:auto">
</body></html>"""


class _Buffer(io.BufferedIOBase):
    def __init__(self):
        self.frame: bytes = b""
        self.ready = threading.Condition()

    def write(self, buf: bytes) -> int:
        with self.ready:
            self.frame = buf
            self.ready.notify_all()
        return len(buf)


_buf = _Buffer()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # silence access log

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(_PAGE)))
            self.end_headers()
            self.wfile.write(_PAGE)

        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    with _buf.ready:
                        _buf.ready.wait()
                        frame = _buf.frame
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except Exception:
                pass
        else:
            self.send_error(404)


cam = Picamera2()
cam.configure(cam.create_video_configuration(
    main={"size": (args.width, args.height), "format": "RGB888"}
))
cam.start_recording(MJPEGEncoder(), FileOutput(_buf))

server = HTTPServer(("0.0.0.0", args.port), _Handler)
print(f"Streaming at http://0.0.0.0:{args.port}  —  Ctrl-C to stop")
try:
    server.serve_forever()
finally:
    cam.stop_recording()
