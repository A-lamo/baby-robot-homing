"""Run an evolved neural-network homing brain on the physical baby robot.

This is the hardware counterpart of
``ariel/examples/re_book/1_brain_evolution_multiprocessing.py``.

It loads the ``best_weights.npy`` (and optionally ``best_weights_meta.npz``)
produced by that script, reconstructs the same MLP network, and drives the
servos at ``--control-hz`` using:

  state = [quat_imag(3)  joint_rads(N)  vision_3sections(3)  phase_sin_cos(2)]

where
  * quat_imag  -- orientation quaternion imaginary parts estimated from IMU acc
  * joint_rads -- servo readback angles converted to radians
  * vision_3   -- green-pixel fraction in left / centre / right camera thirds
  * phase      -- 2*[sin, cos] of elapsed time, matching the training signal

Depends only on numpy and picamera2 (no PyTorch, no OpenCV).

Usage example (Pi):
  python hardware/run_nn_homing.py \\
      --weights __data__/1_brain_evolution_multiprocessing/best_weights.npy \\
      --morphology insect \\
      --duration 60
"""
from __future__ import annotations

import argparse
import io
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
#  Inline MJPEG streaming (optional, enabled with --stream-port)              #
# --------------------------------------------------------------------------- #

class _StreamBuffer:
    """Thread-safe latest-frame store for the MJPEG server."""
    def __init__(self):
        self._frame: bytes = b""
        self._cond  = threading.Condition()

    def push(self, jpeg: bytes) -> None:
        with self._cond:
            self._frame = jpeg
            self._cond.notify_all()

    def wait_frame(self) -> bytes:
        with self._cond:
            self._cond.wait()
            return self._frame


_stream_buf: _StreamBuffer | None = None


def _encode_jpeg(rgb: np.ndarray, quality: int = 70) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _debug_frame(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Dim non-detected pixels and highlight detections in bright green."""
    out = (rgb * 0.25).astype(np.uint8)
    out[mask == 1] = (0, 220, 0)
    return out


class _MJPEGHandler(BaseHTTPRequestHandler):
    _PAGE = (
        b"<!DOCTYPE html><html><head><title>Robot Camera</title></head>"
        b'<body style="margin:0;background:#000">'
        b'<img src="/stream.mjpg" style="max-width:100%;display:block;margin:auto">'
        b"</body></html>"
    )

    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(self._PAGE)))
            self.end_headers()
            self.wfile.write(self._PAGE)
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    assert _stream_buf is not None
                    frame = _stream_buf.wait_frame()
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


def _start_stream_server(port: int) -> None:
    global _stream_buf
    _stream_buf = _StreamBuffer()
    srv = HTTPServer(("0.0.0.0", port), _MJPEGHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    print(f"Stream:      http://0.0.0.0:{port}")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.baby_hardware import (
    BATTERY_CSV_FIELDS,
    BabyRobotHardware,
    ServoMapping,
    append_csv_row,
    battery_sample_row,
    get_servo_mappings_for_morphology,
    make_run_dir,
    write_metadata,
)


# --------------------------------------------------------------------------- #
#  Pure-NumPy MLP (matches Network in 1_brain_evolution_multiprocessing.py)   #
# --------------------------------------------------------------------------- #

def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0.0, x, np.expm1(x))


class NumpyNetwork:
    """3-layer MLP with ELU hidden activations and Tanh output, no PyTorch."""

    def __init__(
        self,
        w1: np.ndarray, b1: np.ndarray,
        w2: np.ndarray, b2: np.ndarray,
        w4: np.ndarray, b4: np.ndarray,
    ) -> None:
        self.w1, self.b1 = w1, b1
        self.w2, self.b2 = w2, b2
        self.w4, self.b4 = w4, b4

    def forward(self, state: np.ndarray) -> np.ndarray:
        x = state.astype(np.float32)
        x = _elu(self.w1 @ x + self.b1)
        x = _elu(self.w2 @ x + self.b2)
        x = np.tanh(self.w4 @ x + self.b4) * (math.pi / 3 * 0.7)
        return x


# --------------------------------------------------------------------------- #
#  Architecture inference                                                      #
# --------------------------------------------------------------------------- #

def _infer_architecture(n: int) -> tuple[int, int, int]:
    """Return (input_dim, hidden_size, output_size) that explains n parameters.

    For fc1(in→h) + fc2(h→h) + fc4(h→out):
    total = h*(in+1) + h*(h+1) + out*(h+1)
    → in = [n - h*(h+2) - out*(h+1)] / h
    """
    for h in (32, 64, 16, 128):
        for out in range(1, 65):
            numerator = n - h * (h + 2) - out * (h + 1)
            if numerator > 0 and numerator % h == 0:
                inp = numerator // h
                if h * (inp + 1) + h * (h + 1) + out * (h + 1) == n:
                    return inp, h, out
    raise ValueError(
        f"Cannot infer architecture from {n} parameters. "
        "Pass --meta to provide an explicit metadata file."
    )


# --------------------------------------------------------------------------- #
#  Loading                                                                     #
# --------------------------------------------------------------------------- #

def load_network(
    weights_path: Path,
    meta_path: Path | None,
) -> tuple[NumpyNetwork, int, int, float]:
    """Return (network, num_joints, control_step_freq, phase_amplitude)."""
    weights = np.load(str(weights_path)).astype(np.float32)

    if meta_path is None:
        candidate = weights_path.parent / (weights_path.stem + "_meta.npz")
        meta_path = candidate if candidate.exists() else None

    if meta_path is not None and meta_path.exists():
        meta = np.load(str(meta_path), allow_pickle=True)
        input_dim    = int(meta["input_dim"])
        hidden_size  = int(meta["hidden_size"])
        output_size  = int(meta["output_size"])
        num_joints   = int(meta["num_joints"])
        control_step_freq = int(meta.get("control_step_freq", 50))
        phase_amplitude   = float(meta.get("phase_amplitude", 2.0))
    else:
        print(
            f"[warn] No metadata file found alongside {weights_path}. "
            "Inferring architecture from weight vector length."
        )
        input_dim, hidden_size, output_size = _infer_architecture(len(weights))
        num_joints        = output_size
        control_step_freq = 50
        phase_amplitude   = 2.0

    # Unpack flat weight vector into layer matrices (PyTorch order: weight, bias per layer)
    ptr = 0
    def _take(shape):
        nonlocal ptr
        n = int(np.prod(shape))
        chunk = weights[ptr : ptr + n].reshape(shape)
        ptr += n
        return chunk

    w1 = _take((hidden_size, input_dim))
    b1 = _take((hidden_size,))
    w2 = _take((hidden_size, hidden_size))
    b2 = _take((hidden_size,))
    w4 = _take((output_size, hidden_size))
    b4 = _take((output_size,))

    if ptr != len(weights):
        raise ValueError(f"Weight vector length {len(weights)} != expected {ptr}")

    net = NumpyNetwork(w1, b1, w2, b2, w4, b4)
    total_params = sum(a.size for a in (w1, b1, w2, b2, w4, b4))
    print(
        f"Loaded network: in={input_dim} hidden={hidden_size} out={output_size} "
        f"({total_params} params)"
    )
    return net, num_joints, control_step_freq, phase_amplitude


# --------------------------------------------------------------------------- #
#  IMU → quaternion imaginary parts                                           #
# --------------------------------------------------------------------------- #

def accel_to_quat_imag(
    acc_x: float,
    acc_y: float,
    acc_z: float,
    z_sign: float = 1.0,
) -> np.ndarray:
    """Estimate quaternion imaginary [x, y, z] from accelerometer gravity vector.

    Assumes the robot is roughly stationary (low-acceleration approximation).
    Yaw is unobservable from the accelerometer and is set to zero.

    ``z_sign``: use +1.0 (default) if the IMU reports acc_z ≈ +9.8 when the
    robot is upright (normal-force convention). Pass -1.0 if acc_z ≈ -9.8
    when upright. When perfectly upright the result should be near [0, 0, 0].
    """
    norm = math.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    if norm < 1e-6:
        return np.zeros(3, dtype=np.float32)
    ax, ay, az = acc_x / norm, acc_y / norm, z_sign * acc_z / norm
    roll  = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2))
    # ZYX Euler → quaternion imaginary parts (yaw = 0)
    cr, sr = math.cos(roll / 2),  math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return np.array([sr * cp, cr * sp, -sr * sp], dtype=np.float32)


# --------------------------------------------------------------------------- #
#  Pure-NumPy vision (green HSV detection, no OpenCV)                         #
# --------------------------------------------------------------------------- #

def _rgb_to_hsv(frame_rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB image to OpenCV-style HSV (H in [0,180], S/V in [0,255])."""
    rgb = frame_rgb.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    diff = cmax - cmin

    # Hue (output in [0, 180] to match OpenCV)
    h = np.zeros_like(r)
    eps = 1e-7
    m_r = (cmax == r) & (diff > eps)
    m_g = (cmax == g) & (diff > eps)
    m_b = (cmax == b) & (diff > eps)
    h[m_r] = (60.0 * ((g[m_r] - b[m_r]) / diff[m_r]) % 360.0) / 2.0
    h[m_g] = (60.0 * ((b[m_g] - r[m_g]) / diff[m_g]) + 120.0) / 2.0
    h[m_b] = (60.0 * ((r[m_b] - g[m_b]) / diff[m_b]) + 240.0) / 2.0

    # Saturation [0, 255]
    s = np.where(cmax > eps, diff / cmax * 255.0, 0.0)

    # Value [0, 255]
    v = cmax * 255.0

    return np.stack([h, s, v], axis=-1)


def isolate_color(frame_rgb: np.ndarray, lower_hsv, upper_hsv) -> np.ndarray:
    """Return a binary uint8 mask for pixels inside the HSV range."""
    hsv = _rgb_to_hsv(frame_rgb)
    lo = np.asarray(lower_hsv, dtype=np.float32)
    hi = np.asarray(upper_hsv, dtype=np.float32)
    mask = np.all((hsv >= lo) & (hsv <= hi), axis=-1)
    return mask.astype(np.uint8)


def analyze_sections_3(mask: np.ndarray) -> list[float]:
    """Return green fraction in left / centre / right thirds (matches training)."""
    sections = np.array_split(mask, 3, axis=1)
    return [float(s.sum()) / s.size if s.size > 0 else 0.0 for s in sections]


# --------------------------------------------------------------------------- #
#  Camera (picamera2 only, no OpenCV)                                         #
# --------------------------------------------------------------------------- #

def open_camera(width: int, height: int):
    from picamera2 import Picamera2
    cam = Picamera2()
    cfg = cam.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"}
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(0.5)
    return cam


# --------------------------------------------------------------------------- #
#  Servo readback → joint radians                                              #
# --------------------------------------------------------------------------- #

def read_joint_radians(robot: BabyRobotHardware) -> np.ndarray:
    return np.array(
        [
            robot.mappings[i].sign * math.radians(
                robot.read_servo_degrees(i) - robot.mappings[i].neutral_deg
            )
            for i in range(len(robot.mappings))
        ],
        dtype=np.float32,
    )


# --------------------------------------------------------------------------- #
#  State vector assembly                                                       #
# --------------------------------------------------------------------------- #

def build_state(
    quat_imag: np.ndarray,
    joint_rads: np.ndarray,
    vision: list[float],
    elapsed_s: float,
    phase_amplitude: float,
) -> np.ndarray:
    phase = [
        phase_amplitude * math.sin(elapsed_s * 2.0 * math.pi),
        phase_amplitude * math.cos(elapsed_s * 2.0 * math.pi),
    ]
    return np.concatenate([quat_imag, joint_rads, vision, phase]).astype(np.float32)


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an evolved neural-network homing brain on the baby robot."
    )
    parser.add_argument(
        "--weights", type=Path, required=True,
        help="Path to best_weights.npy from training.",
    )
    parser.add_argument(
        "--meta", type=Path, default=None,
        help="Path to best_weights_meta.npz. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--morphology", default="insect",
        choices=["baby", "insect"],
        help="Servo mapping preset matching the trained morphology.",
    )
    parser.add_argument(
        "--duration", type=float, default=0.0,
        help="Seconds to run. 0 = run until Ctrl-C.",
    )
    parser.add_argument(
        "--control-hz", type=float, default=None,
        help="Control loop frequency in Hz. Defaults to control_step_freq from metadata.",
    )
    parser.add_argument("--width",  type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument(
        "--camera-order", choices=["rgb", "bgr"], default="rgb",
        help="Pixel channel order from Picamera2 (RGB888 format → rgb).",
    )
    # HSV defaults tuned for a bright-green target matching the simulation colour.
    parser.add_argument("--lower-hsv", type=int, nargs=3, default=(35, 40, 40))
    parser.add_argument("--upper-hsv", type=int, nargs=3, default=(85, 255, 255))
    parser.add_argument(
        "--imu-z-sign", type=float, default=1.0, choices=[1.0, -1.0],
        help="+1 if acc_z ≈ +9.8 when upright, -1 if acc_z ≈ -9.8.",
    )
    parser.add_argument("--neutral-hold-s", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-beep",    action="store_true")
    parser.add_argument("--delay-mode", action="store_true")
    parser.add_argument(
        "--stream-port", type=int, default=0,
        help="Start an MJPEG stream on this port (e.g. 8000). 0 = disabled.",
    )
    parser.add_argument(
        "--debug-vision", action="store_true",
        help="Stream the green detection mask instead of the raw frame (requires --stream-port).",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
#  CSV logging                                                                 #
# --------------------------------------------------------------------------- #

def _csv_fields(num_joints: int, num_servos: int) -> list[str]:
    return [
        "elapsed_s", "loop_dt_s",
        "quat_imag_x", "quat_imag_y", "quat_imag_z",
        *[f"joint_{i}_rad"       for i in range(num_joints)],
        "vision_left", "vision_centre", "vision_right",
        "phase_sin", "phase_cos",
        *[f"action_{i}_rad"      for i in range(num_joints)],
        *[f"servo_{i}_command_deg" for i in range(num_servos)],
    ]


def _make_row(
    elapsed_s, loop_dt_s, quat_imag, joint_rads, vision, state,
    action, robot, num_joints, phase_offset,
) -> dict:
    row: dict = {
        "elapsed_s":    f"{elapsed_s:.6f}",
        "loop_dt_s":    f"{loop_dt_s:.6f}",
        "quat_imag_x":  f"{quat_imag[0]:.6f}",
        "quat_imag_y":  f"{quat_imag[1]:.6f}",
        "quat_imag_z":  f"{quat_imag[2]:.6f}",
        "vision_left":   f"{vision[0]:.6f}",
        "vision_centre": f"{vision[1]:.6f}",
        "vision_right":  f"{vision[2]:.6f}",
        "phase_sin":    f"{state[phase_offset]:.6f}",
        "phase_cos":    f"{state[phase_offset + 1]:.6f}",
    }
    for i in range(num_joints):
        row[f"joint_{i}_rad"]  = f"{joint_rads[i]:.6f}"
        row[f"action_{i}_rad"] = f"{action[i]:.6f}"
    for i, deg in enumerate(robot.last_command_degrees):
        row[f"servo_{i}_command_deg"] = f"{deg:.6f}"
    return row


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()

    net, num_joints, control_step_freq, phase_amplitude = load_network(
        args.weights.resolve(), args.meta
    )
    control_hz     = args.control_hz if args.control_hz is not None else float(control_step_freq)
    control_period = 1.0 / control_hz
    phase_offset   = 3 + num_joints + 3  # position of phase slice in state vector

    mappings = get_servo_mappings_for_morphology(args.morphology)
    if len(mappings) != num_joints:
        print(
            f"[warn] '{args.morphology}' has {len(mappings)} servo mappings "
            f"but network expects {num_joints} joints. Check --morphology."
        )

    run_dir     = make_run_dir("run_nn_homing", args.output_dir)
    nn_csv      = run_dir / "nn_samples.csv"
    battery_csv = run_dir / "battery_samples.csv"
    csv_fields  = _csv_fields(num_joints, len(mappings))

    if args.stream_port:
        _start_stream_server(args.stream_port)

    camera = open_camera(args.width, args.height)
    robot  = BabyRobotHardware(mappings=mappings, beep=not args.no_beep, direct_mode=not args.delay_mode)

    print(f"Results:     {run_dir}")
    print(f"Morphology:  {args.morphology}  joints={num_joints}  hz={control_hz:.1f}")
    print(f"HSV range:   lower={tuple(args.lower_hsv)} upper={tuple(args.upper_hsv)}")
    print(f"Duration:    {'∞ (Ctrl-C to stop)' if args.duration <= 0 else f'{args.duration:.1f}s'}")

    write_metadata(run_dir / "run_meta.json", {
        "weights": str(args.weights),
        "morphology": args.morphology,
        "num_joints": num_joints,
        "control_hz": control_hz,
        "phase_amplitude": phase_amplitude,
        "lower_hsv": list(args.lower_hsv),
        "upper_hsv": list(args.upper_hsv),
        "duration": args.duration,
    })

    robot.neutral()
    time.sleep(args.neutral_hold_s)

    battery_samples: list = []
    start_time      = time.monotonic()
    prev_loop_time  = start_time
    loop_count      = 0

    print("Running… (Ctrl-C to stop)")
    try:
        while True:
            loop_start = time.monotonic()
            elapsed_s  = loop_start - start_time

            if args.duration > 0 and elapsed_s >= args.duration:
                break

            # Vision
            raw = camera.capture_array()
            if args.camera_order == "bgr":
                raw = raw[..., ::-1]          # BGR → RGB (no cv2 needed)
            mask   = isolate_color(raw, args.lower_hsv, args.upper_hsv)
            vision = analyze_sections_3(mask)
            if _stream_buf is not None:
                frame = _debug_frame(raw, mask) if args.debug_vision else raw
                _stream_buf.push(_encode_jpeg(frame))

            # Orientation
            imu       = robot.read_imu(start_time)
            quat_imag = accel_to_quat_imag(imu.acc_x, imu.acc_y, imu.acc_z, z_sign=args.imu_z_sign)

            # Joint positions
            joint_rads = read_joint_radians(robot)

            # Inference
            state  = build_state(quat_imag, joint_rads, vision, elapsed_s, phase_amplitude)
            action = net.forward(state)

            # Actuate
            robot.set_joint_angles(action)

            # Logging
            battery = robot.read_battery(start_time, battery_samples)
            append_csv_row(battery_csv, BATTERY_CSV_FIELDS, battery_sample_row(battery))
            loop_dt_s      = loop_start - prev_loop_time
            prev_loop_time = loop_start
            append_csv_row(
                nn_csv, csv_fields,
                _make_row(elapsed_s, loop_dt_s, quat_imag, joint_rads, vision, state, action, robot, num_joints, phase_offset),
            )

            loop_count += 1
            if loop_count % 50 == 0:
                fps = loop_count / elapsed_s if elapsed_s > 0 else 0.0
                print(
                    f"t={elapsed_s:6.1f}s  fps={fps:.1f}  "
                    f"vis=[{vision[0]:.3f},{vision[1]:.3f},{vision[2]:.3f}]  "
                    f"battery={battery.percentage}%"
                )

            time.sleep(max(0.0, control_period - (time.monotonic() - loop_start)))

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        print(f"Returning to neutral for {args.neutral_hold_s:.1f}s.")
        robot.neutral()
        time.sleep(args.neutral_hold_s)
        robot.close()
        camera.stop()

    elapsed_total = time.monotonic() - start_time
    actual_hz     = loop_count / elapsed_total if elapsed_total > 0 else 0.0
    print(f"Done. {loop_count} steps in {elapsed_total:.1f}s ({actual_hz:.1f} Hz avg)")
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
