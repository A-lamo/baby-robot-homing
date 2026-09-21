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

If no metadata file is found the architecture is inferred from the weight
vector length.

Usage example (Pi):
  python hardware/run_nn_homing.py \\
      --weights __data__/1_brain_evolution_multiprocessing/best_weights.npy \\
      --morphology insect \\
      --duration 60
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

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
from robot_control.vision import open_camera


# --------------------------------------------------------------------------- #
#  Network (must match the architecture in 1_brain_evolution_multiprocessing) #
# --------------------------------------------------------------------------- #

class Network(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.hidden_activation = nn.ELU()
        self.output_activation = nn.Tanh()
        for param in self.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def forward(self, state: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(state, dtype=torch.float32)
        x = self.hidden_activation(self.fc1(x))
        x = self.hidden_activation(self.fc2(x))
        x = self.output_activation(self.fc4(x)) * (torch.pi / 2)
        return x.numpy()


def _fill_parameters(net: nn.Module, vector: np.ndarray) -> None:
    """Load a flat weight vector into a network in-place."""
    ptr = 0
    for p in net.parameters():
        n = p.numel()
        p.data.copy_(torch.as_tensor(vector[ptr : ptr + n]).view(p.shape))
        ptr += n
    if ptr != len(vector):
        raise ValueError(f"Weight vector length {len(vector)} != network params {ptr}")


# --------------------------------------------------------------------------- #
#  Architecture inference from weight vector length                           #
# --------------------------------------------------------------------------- #

def _infer_architecture(n: int) -> tuple[int, int, int]:
    """Return (input_dim, hidden_size, output_size) that explains n parameters.

    For the 3-layer network: fc1(in→h) + fc2(h→h) + fc4(h→out)
    total = h*(in+1) + h*(h+1) + out*(h+1)
    Rearranging: in = [n - h*(h+2) - out*(h+1)] / h
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
) -> tuple[Network, int, int, float, int]:
    """Return (network, num_joints, control_step_freq, phase_amplitude, vision_sections)."""
    weights = np.load(str(weights_path))

    if meta_path is None:
        candidate = weights_path.parent / (weights_path.stem + "_meta.npz")
        meta_path = candidate if candidate.exists() else None

    if meta_path is not None and meta_path.exists():
        meta = np.load(str(meta_path), allow_pickle=True)
        input_dim = int(meta["input_dim"])
        hidden_size = int(meta["hidden_size"])
        output_size = int(meta["output_size"])
        num_joints = int(meta["num_joints"])
        control_step_freq = int(meta.get("control_step_freq", 50))
        phase_amplitude = float(meta.get("phase_amplitude", 2.0))
        vision_sections = int(meta.get("vision_sections", 3))
    else:
        print(
            f"[warn] No metadata file found alongside {weights_path}. "
            "Inferring architecture from weight vector length."
        )
        input_dim, hidden_size, output_size = _infer_architecture(len(weights))
        num_joints = output_size
        control_step_freq = 50
        phase_amplitude = 2.0
        vision_sections = 3

    net = Network(input_size=input_dim, output_size=output_size, hidden_size=hidden_size)
    _fill_parameters(net, weights)
    net.eval()

    print(
        f"Loaded network: in={input_dim} hidden={hidden_size} out={output_size} "
        f"({sum(p.numel() for p in net.parameters())} params)"
    )
    return net, num_joints, control_step_freq, phase_amplitude, vision_sections


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
    robot is upright (Z-up convention, normal force pointing up).  Pass -1.0
    if your board reports acc_z ≈ -9.8 when upright (gravity convention).
    When the robot is perfectly upright the result should be near [0, 0, 0].
    """
    norm = math.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    if norm < 1e-6:
        return np.zeros(3, dtype=np.float32)
    ax, ay, az = acc_x / norm, acc_y / norm, z_sign * acc_z / norm
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2))
    # ZYX Euler → quaternion imaginary parts (yaw = 0, so cy=1, sy=0)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return np.array([sr * cp, cr * sp, -sr * sp], dtype=np.float32)


# --------------------------------------------------------------------------- #
#  Vision                                                                     #
# --------------------------------------------------------------------------- #

def isolate_color(frame_rgb: np.ndarray, lower_hsv, upper_hsv) -> np.ndarray:
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    return cv2.inRange(
        hsv,
        np.asarray(lower_hsv, dtype=np.uint8),
        np.asarray(upper_hsv, dtype=np.uint8),
    )


def analyze_sections_3(mask: np.ndarray) -> list[float]:
    """Return green fraction in left / centre / right thirds (matches training)."""
    sections = np.array_split(mask, 3, axis=1)
    return [
        float(cv2.countNonZero(s)) / s.size if s.size > 0 else 0.0
        for s in sections
    ]


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
#  Servo readback → joint radians                                              #
# --------------------------------------------------------------------------- #

def servo_degrees_to_rad(degrees: float, mapping: ServoMapping) -> float:
    return mapping.sign * math.radians(degrees - mapping.neutral_deg)


def read_joint_radians(robot: BabyRobotHardware) -> np.ndarray:
    return np.array(
        [
            servo_degrees_to_rad(robot.read_servo_degrees(i), robot.mappings[i])
            for i in range(len(robot.mappings))
        ],
        dtype=np.float32,
    )


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
        choices=list({"baby", "insect"}),
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
    parser.add_argument(
        "--width", type=int, default=320,
        help="Camera capture width in pixels.",
    )
    parser.add_argument(
        "--height", type=int, default=240,
        help="Camera capture height in pixels.",
    )
    parser.add_argument(
        "--camera-order", choices=["rgb", "bgr"], default="bgr",
        help="Pixel channel order from Picamera2 (usually bgr).",
    )
    # HSV defaults tuned for a bright-green target similar to the simulation.
    # Adjust with --lower-hsv / --upper-hsv if your target colour differs.
    parser.add_argument("--lower-hsv", type=int, nargs=3, default=(35, 40, 40))
    parser.add_argument("--upper-hsv", type=int, nargs=3, default=(85, 255, 255))
    parser.add_argument(
        "--neutral-hold-s", type=float, default=1.0,
        help="Seconds to hold neutral pose before and after the run.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override the default results output directory.",
    )
    parser.add_argument(
        "--imu-z-sign", type=float, default=1.0, choices=[1.0, -1.0],
        help=(
            "Sign of the IMU Z axis. +1 (default) if acc_z ≈ +9.8 when upright "
            "(normal-force convention). -1 if acc_z ≈ -9.8 when upright."
        ),
    )
    parser.add_argument(
        "--no-beep", action="store_true",
        help="Skip the startup beep.",
    )
    parser.add_argument(
        "--delay-mode", action="store_true",
        help="Use delayed servo mode (smoother, less responsive).",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
#  CSV logging                                                                 #
# --------------------------------------------------------------------------- #

NN_CSV_FIELDS = [
    "elapsed_s",
    "loop_dt_s",
    *[f"quat_imag_{c}" for c in ("x", "y", "z")],
    *[f"joint_{i}_rad" for i in range(8)],
    "vision_left",
    "vision_centre",
    "vision_right",
    "phase_sin",
    "phase_cos",
    *[f"action_{i}_rad" for i in range(8)],
    *[f"servo_{i}_command_deg" for i in range(8)],
]


def make_nn_row(
    elapsed_s: float,
    loop_dt_s: float,
    quat_imag: np.ndarray,
    joint_rads: np.ndarray,
    vision: list[float],
    state: np.ndarray,
    action: np.ndarray,
    robot: BabyRobotHardware,
    num_joints: int,
) -> dict:
    phase_offset = 3 + num_joints + 3  # position of phase slice in state
    row: dict = {
        "elapsed_s": f"{elapsed_s:.6f}",
        "loop_dt_s": f"{loop_dt_s:.6f}",
        "quat_imag_x": f"{quat_imag[0]:.6f}",
        "quat_imag_y": f"{quat_imag[1]:.6f}",
        "quat_imag_z": f"{quat_imag[2]:.6f}",
        "vision_left": f"{vision[0]:.6f}",
        "vision_centre": f"{vision[1]:.6f}",
        "vision_right": f"{vision[2]:.6f}",
        "phase_sin": f"{state[phase_offset]:.6f}",
        "phase_cos": f"{state[phase_offset + 1]:.6f}",
    }
    for i in range(num_joints):
        row[f"joint_{i}_rad"] = f"{joint_rads[i]:.6f}"
    for i in range(num_joints):
        row[f"action_{i}_rad"] = f"{action[i]:.6f}" if i < len(action) else "0.000000"
    for i in range(8):
        row[f"servo_{i}_command_deg"] = (
            f"{robot.last_command_degrees[i]:.6f}" if i < len(robot.last_command_degrees) else "90.000000"
        )
    return row


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()

    net, num_joints, control_step_freq, phase_amplitude, _ = load_network(
        args.weights.resolve(), args.meta
    )
    control_hz = args.control_hz if args.control_hz is not None else float(control_step_freq)
    control_period_s = 1.0 / control_hz

    mappings = get_servo_mappings_for_morphology(args.morphology)
    if len(mappings) != num_joints:
        print(
            f"[warn] Morphology '{args.morphology}' has {len(mappings)} servo mappings "
            f"but network expects {num_joints} joints. Check --morphology."
        )

    run_dir = make_run_dir("run_nn_homing", args.output_dir)
    nn_csv = run_dir / "nn_samples.csv"
    battery_csv = run_dir / "battery_samples.csv"

    robot = BabyRobotHardware(
        mappings=mappings,
        beep=not args.no_beep,
        direct_mode=not args.delay_mode,
    )
    camera = open_camera(args.width, args.height)

    print(f"Results: {run_dir}")
    print(f"Morphology: {args.morphology}  Joints: {num_joints}  Control Hz: {control_hz:.1f}")
    print(f"HSV range: lower={tuple(args.lower_hsv)} upper={tuple(args.upper_hsv)}")
    if args.duration > 0:
        print(f"Duration: {args.duration:.1f}s")
    else:
        print("Duration: run until Ctrl-C")

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
    start_time = time.monotonic()
    prev_loop_time = start_time
    loop_count = 0

    print("Running neural network homing brain… (Ctrl-C to stop)")
    try:
        while True:
            loop_start = time.monotonic()
            elapsed_s = loop_start - start_time

            if args.duration > 0 and elapsed_s >= args.duration:
                break

            # --- Vision ---
            raw_frame = camera.capture_array()
            frame_rgb = raw_frame if args.camera_order == "rgb" else cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            mask = isolate_color(frame_rgb, args.lower_hsv, args.upper_hsv)
            vision = analyze_sections_3(mask)

            # --- Orientation (IMU accelerometer → quaternion imaginary) ---
            imu = robot.read_imu(start_time)
            quat_imag = accel_to_quat_imag(imu.acc_x, imu.acc_y, imu.acc_z, z_sign=args.imu_z_sign)

            # --- Joint positions ---
            joint_rads = read_joint_radians(robot)

            # --- State & inference ---
            state = build_state(quat_imag, joint_rads, vision, elapsed_s, phase_amplitude)
            action = net.forward(state)

            # --- Actuate ---
            robot.set_joint_angles(action)

            # --- Battery & logging ---
            battery = robot.read_battery(start_time, battery_samples)
            append_csv_row(battery_csv, BATTERY_CSV_FIELDS, battery_sample_row(battery))

            loop_dt_s = loop_start - prev_loop_time
            prev_loop_time = loop_start

            # Build CSV row with dynamic joint count
            row = make_nn_row(
                elapsed_s, loop_dt_s, quat_imag, joint_rads, vision, state, action, robot, num_joints
            )
            # Prune CSV fields to actual joint count to avoid empty columns
            fields = [
                "elapsed_s", "loop_dt_s",
                *[f"quat_imag_{c}" for c in ("x", "y", "z")],
                *[f"joint_{i}_rad" for i in range(num_joints)],
                "vision_left", "vision_centre", "vision_right",
                "phase_sin", "phase_cos",
                *[f"action_{i}_rad" for i in range(num_joints)],
                *[f"servo_{i}_command_deg" for i in range(len(mappings))],
            ]
            append_csv_row(nn_csv, fields, row)

            loop_count += 1
            if loop_count % 50 == 0:
                fps = loop_count / elapsed_s if elapsed_s > 0 else 0.0
                print(
                    f"t={elapsed_s:6.1f}s  fps={fps:.1f}  "
                    f"vision=[{vision[0]:.3f},{vision[1]:.3f},{vision[2]:.3f}]  "
                    f"battery={battery.percentage}%"
                )

            time.sleep(max(0.0, control_period_s - (time.monotonic() - loop_start)))

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        print(f"Returning to neutral for {args.neutral_hold_s:.1f}s.")
        robot.neutral()
        time.sleep(args.neutral_hold_s)
        robot.close()
        camera.stop()

    elapsed_total = time.monotonic() - start_time
    actual_hz = loop_count / elapsed_total if elapsed_total > 0 else 0.0
    print(f"Done. {loop_count} steps in {elapsed_total:.1f}s ({actual_hz:.1f} Hz avg)")
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
