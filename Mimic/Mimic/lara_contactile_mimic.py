#!/usr/bin/env python3
"""Control a NEURA LARA and Contactile gripper with a laptop camera.

The camera is mirrored. A thumb/index pinch acts as a dead-man clutch for TCP
motion. An open palm opens the gripper and a fist closes it. Real hardware is
only addressed when --execute is supplied. Control is armed automatically.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import cv2
except ImportError:  # Keep --help and unit tests usable without camera packages.
    cv2 = None

try:
    import mediapipe as mp
except ImportError:
    mp = None


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def install_cleanup_signal_handlers() -> None:
    """Turn terminal stop/termination signals into a clean Python shutdown."""

    def request_shutdown(signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        print(
            f"\n{signal_name} empfangen: LARA wird gestoppt und die Kamera freigegeben.",
            file=sys.stderr,
        )
        raise KeyboardInterrupt

    for signal_name in ("SIGTSTP", "SIGTERM", "SIGHUP", "SIGQUIT"):
        shutdown_signal = getattr(signal, signal_name, None)
        if shutdown_signal is not None:
            signal.signal(shutdown_signal, request_shutdown)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_normalize(quaternion: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) ** 2 for value in quaternion))
    if norm <= 1e-9:
        raise ValueError("Quaternion has zero length")
    return [float(value) / norm for value in quaternion]


def quaternion_multiply(left: Sequence[float], right: Sequence[float]) -> list[float]:
    """Multiply two quaternions in the LARA wxyz convention."""
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return quaternion_normalize(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def axis_angle_quaternion(axis: str, angle: float) -> list[float]:
    half = 0.5 * angle
    sine = math.sin(half)
    components = {"x": (sine, 0.0, 0.0), "y": (0.0, sine, 0.0), "z": (0.0, 0.0, sine)}
    return [math.cos(half), *components[axis]]


def quaternion_slerp(
    start: Sequence[float], target: Sequence[float], fraction: float
) -> list[float]:
    start_q = quaternion_normalize(start)
    target_q = quaternion_normalize(target)
    dot = sum(a * b for a, b in zip(start_q, target_q))
    if dot < 0.0:
        target_q = [-value for value in target_q]
        dot = -dot
    dot = clamp(dot, -1.0, 1.0)
    fraction = clamp(fraction, 0.0, 1.0)
    if dot > 0.9995:
        return quaternion_normalize(
            [start_q[index] + fraction * (target_q[index] - start_q[index]) for index in range(4)]
        )
    theta = math.acos(dot)
    sine = math.sin(theta)
    start_weight = math.sin((1.0 - fraction) * theta) / sine
    target_weight = math.sin(fraction * theta) / sine
    return quaternion_normalize(
        [start_weight * start_q[index] + target_weight * target_q[index] for index in range(4)]
    )


def distance_2d(point_a: Any, point_b: Any) -> float:
    return math.hypot(point_a.x - point_b.x, point_a.y - point_b.y)


def palm_center(landmarks: Sequence[Any]) -> tuple[float, float]:
    weighted_points = (
        (0, 0.18),
        (1, 0.08),
        (5, 0.17),
        (9, 0.22),
        (13, 0.18),
        (17, 0.17),
    )
    return (
        sum(landmarks[index].x * weight for index, weight in weighted_points),
        sum(landmarks[index].y * weight for index, weight in weighted_points),
    )


def palm_roll_angle(landmarks: Sequence[Any]) -> float:
    """Return the visible palm rotation in the camera image."""
    return math.atan2(
        landmarks[5].y - landmarks[17].y,
        landmarks[5].x - landmarks[17].x,
    )


def finger_extended(
    landmarks: Sequence[Any],
    tip_index: int,
    pip_index: int,
    center: tuple[float, float],
) -> bool:
    tip_distance = math.hypot(
        landmarks[tip_index].x - center[0], landmarks[tip_index].y - center[1]
    )
    pip_distance = math.hypot(
        landmarks[pip_index].x - center[0], landmarks[pip_index].y - center[1]
    )
    return tip_distance > pip_distance * 1.25


def detect_gripper_gesture(
    landmarks: Sequence[Any], pinch_active: bool
) -> str | None:
    if pinch_active:
        return None
    center = palm_center(landmarks)
    extended = sum(
        finger_extended(landmarks, tip, pip, center)
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
    )
    if extended >= 4:
        return "OPEN"
    if extended <= 1:
        return "CLOSE"
    return None


def map_hand_delta(
    base_position: Sequence[float],
    horizontal_delta: float,
    vertical_delta: float,
    depth_delta: float,
    args: argparse.Namespace,
) -> list[float]:
    """Map mirrored laptop-camera motion to LARA base-frame XYZ (metres)."""
    position = [float(value) for value in base_position[:3]]
    movements = (
        (
            args.horizontal_axis,
            clamp(horizontal_delta * args.horizontal_scale, -args.max_horizontal, args.max_horizontal)
            * args.horizontal_sign,
        ),
        (
            args.vertical_axis,
            clamp(-vertical_delta * args.vertical_scale, -args.max_vertical, args.max_vertical)
            * args.vertical_sign,
        ),
        (
            args.depth_axis,
            clamp(depth_delta * args.depth_scale, -args.max_depth, args.max_depth)
            * args.depth_sign,
        ),
    )
    for axis, movement in movements:
        position[AXIS_INDEX[axis]] += movement
    return position


def map_hand_rotation(
    anchor_quaternion: Sequence[float],
    horizontal_delta: float,
    vertical_delta: float,
    roll_delta: float,
    args: argparse.Namespace,
) -> list[float]:
    """Map rotation-mode hand motion to a base-frame quaternion (wxyz)."""
    limit = math.radians(args.max_rotation_deg)
    yaw = clamp(horizontal_delta * args.rotation_scale, -limit, limit)
    pitch = clamp(-vertical_delta * args.rotation_scale, -limit, limit)
    roll = clamp(roll_delta * args.roll_scale, -limit, limit)
    delta = quaternion_multiply(
        axis_angle_quaternion("z", yaw),
        quaternion_multiply(
            axis_angle_quaternion("x", pitch),
            axis_angle_quaternion("y", roll),
        ),
    )
    return quaternion_multiply(delta, anchor_quaternion)


def discover_gripper_ports(requested_port: str) -> list[str]:
    if requested_port != "auto":
        if requested_port.startswith("/dev/"):
            requested_port = os.path.basename(os.path.realpath(requested_port))
        return [requested_port.removeprefix("tty")]

    devices = sorted(glob.glob("/dev/ttyACM*"))
    devices += sorted(glob.glob("/dev/ttyUSB*"))
    devices += sorted(glob.glob("/dev/serial/by-id/*"))
    ports = []
    for device in devices:
        name = os.path.basename(os.path.realpath(device)).removeprefix("tty")
        if name not in ports:
            ports.append(name)
    if not ports:
        raise RuntimeError(
            "Kein Contactile-USB-Geraet gefunden. Kabel/Stromversorgung und "
            "/dev/ttyACM*, /dev/ttyUSB* sowie Benutzerrechte pruefen."
        )
    return ports


class ContactileController:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = None
        self.state: str | None = None

    def connect(self) -> None:
        try:
            from contactile_gripper_lib.contactile_gripper_lib import GripperClass
        except ImportError:
            try:
                from contactile_gripper_lib import GripperClass
            except ImportError as exc:
                raise RuntimeError(
                    "contactile_gripper_lib ist nicht installiert. Die Herstellerbibliothek "
                    "in die aktive Python-Umgebung installieren."
                ) from exc

        errors = []
        for port in discover_gripper_ports(self.args.gripper_port):
            candidate = GripperClass(False)
            print(f"Contactile: versuche Port {port} ...")
            if not candidate.setComPort(port):
                errors.append(port)
                continue
            deadline = time.monotonic() + self.args.gripper_ready_timeout
            while time.monotonic() < deadline:
                if candidate.sendReceiveReady():
                    self.device = candidate
                    print(f"Contactile verbunden: {port}")
                    print("Contactile CLEAR_ERROR_STATE:", candidate.clear_error_state())
                    if self.args.tactile_stream:
                        print("Contactile DATA_STREAM:", candidate.data_stream())
                        print("Contactile BIAS:", candidate.bias())
                    return
                time.sleep(0.05)
            errors.append(f"{port} (nicht ready)")
        raise RuntimeError("Contactile-Verbindung fehlgeschlagen: " + ", ".join(errors))

    def command(self, state: str) -> None:
        if state == self.state:
            return
        if self.device is None:
            print(f"DRY-RUN Contactile: {state}")
            self.state = state
            return
        if state == "OPEN":
            print("Contactile OPEN:", self.device.release())
            time.sleep(self.args.release_wait)
            print(
                f"Contactile WIDTH {self.args.open_width:.1f} mm:",
                self.device.pc_move_to_width(self.args.open_width),
            )
        elif state == "CLOSE":
            print("Contactile RELEASE POSITION MODE:", self.device.release())
            time.sleep(self.args.release_wait)
            print(
                f"Contactile WIDTH {self.args.closed_width:.1f} mm:",
                self.device.pc_move_to_width(self.args.closed_width),
            )
        else:
            raise ValueError(f"Unbekannter Gripper-Zustand: {state}")
        self.state = state


@dataclass
class MotionResult:
    pose: list[float]
    workspace_limited: bool


class LaraController:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.robot = None
        self.armed = False
        self.commanded_pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        self.workspace_center = [0.0, 0.0, 0.0]
        self.last_servo_response: Any = None
        self.last_command_time = time.monotonic()

    def connect(self) -> None:
        if not self.args.execute:
            print("Kamera-Testmodus: LARA und Contactile werden nicht angesprochen.")
            return
        try:
            from neurapy.robot import Robot
        except ImportError as exc:
            raise RuntimeError(
                "neurapy ist nicht installiert. Bitte die zum LARA-Controller passende "
                "NEURA-Pythonbibliothek in die aktive Umgebung installieren."
            ) from exc
        self.robot = Robot(socket_address=self.args.robot_ip)
        self._read_pose_as_reference()
        print(f"LARA verbunden: {self.args.robot_ip}")

    def _read_pose_as_reference(self) -> None:
        if self.robot is not None:
            if not hasattr(self.robot, "get_tcp_pose_quaternion"):
                raise RuntimeError(
                    "Die installierte neurapy-Version bietet get_tcp_pose_quaternion() nicht an."
                )
            pose = self.robot.get_tcp_pose_quaternion()
            if not pose or len(pose) < 7:
                raise RuntimeError(f"Ungueltige LARA TCP-Pose: {pose!r}")
            self.commanded_pose = [float(value) for value in pose[:7]]
        self.workspace_center = self.commanded_pose[:3]
        self.last_command_time = time.monotonic()

    def set_armed(self, armed: bool) -> None:
        if armed == self.armed:
            return
        if armed:
            try:
                if self.robot is not None:
                    self.robot.init_program()
                    self._read_pose_as_reference()
                    activated = self.robot.activate_servo_interface("position")
                    print(f"LARA Servo-Aktivierung: {activated}")
                    if activated is False:
                        print(
                            "WARNUNG: activate_servo_interface() meldet False. Wegen der "
                            "bekannten NeuraPy-Versionsabweichung wird im Kompatibilitaetsmodus "
                            "fortgefahren; ein echter servo_x/servo_j-Fehler stoppt das Skript."
                        )
                else:
                    self._read_pose_as_reference()
            except Exception:
                if self.robot is not None:
                    try:
                        self.robot.deactivate_servo_interface()
                    except Exception:
                        pass
                    try:
                        self.robot.stop()
                    except Exception:
                        pass
                raise
            self.armed = True
            print("AKTIV: Pinches bewegen/rotieren LARA; Handgesten steuern Contactile.")
        else:
            self.armed = False
            if self.robot is not None:
                try:
                    self.robot.deactivate_servo_interface()
                except Exception as exc:
                    print(f"Hinweis beim Deaktivieren der Servo-Schnittstelle: {exc}")
                try:
                    self.robot.stop()
                except Exception as exc:
                    print(f"Hinweis beim LARA-Stop: {exc}")
            print("INAKTIV: keine Bewegungs- oder Gripperbefehle.")

    def current_pose(self) -> list[float]:
        if self.robot is not None:
            pose = self.robot.get_tcp_pose_quaternion()
            if pose and len(pose) >= 7:
                self.commanded_pose = [float(value) for value in pose[:7]]
        return list(self.commanded_pose)

    def _safe_and_limited_pose(self, desired_pose: Sequence[float]) -> MotionResult:
        bounds = (self.args.workspace_x, self.args.workspace_y, self.args.workspace_z)
        desired = [float(value) for value in desired_pose[:7]]
        workspace_limited = False
        for index, radius in enumerate(bounds):
            low = self.workspace_center[index] - radius
            high = self.workspace_center[index] + radius
            limited = clamp(desired[index], low, high)
            workspace_limited |= not math.isclose(limited, desired[index], abs_tol=1e-9)
            desired[index] = limited

        dt = max(0.001, time.monotonic() - self.last_command_time)
        max_step = min(self.args.max_tcp_step, self.args.max_speed * dt)
        delta = [
            (desired[index] - self.commanded_pose[index]) * self.args.target_smoothing
            for index in range(3)
        ]
        distance = math.sqrt(sum(value * value for value in delta))
        if distance > max_step:
            delta = [value * max_step / distance for value in delta]
        pose = list(self.commanded_pose)
        pose[:3] = [self.commanded_pose[index] + delta[index] for index in range(3)]
        current_q = quaternion_normalize(self.commanded_pose[3:7])
        desired_q = quaternion_normalize(desired[3:7])
        quaternion_dot = abs(sum(a * b for a, b in zip(current_q, desired_q)))
        angular_distance = 2.0 * math.acos(clamp(quaternion_dot, -1.0, 1.0))
        max_angular_step = min(
            math.radians(self.args.max_angular_step_deg),
            self.args.max_angular_speed * dt,
        )
        orientation_fraction = self.args.orientation_smoothing
        if angular_distance > 1e-9:
            orientation_fraction = min(
                orientation_fraction,
                max_angular_step / angular_distance,
            )
        pose[3:7] = quaternion_slerp(current_q, desired_q, orientation_fraction)
        return MotionResult(pose, workspace_limited)

    def send_pose(self, desired_pose: Sequence[float]) -> MotionResult:
        result = self._safe_and_limited_pose(desired_pose)
        now = time.monotonic()
        dt = max(0.001, now - self.last_command_time)
        velocity = [
            (result.pose[index] - self.commanded_pose[index]) / dt for index in range(3)
        ] + [0.0, 0.0, 0.0]
        if self.robot is not None:
            response = self.robot.servo_x(
                result.pose,
                velocity,
                [0.0] * 6,
                self.args.servo_gain,
            )
            if isinstance(response, (int, float)) and response >= 3:
                raise RuntimeError(f"LARA servo_x Fehlercode: {response}")
            self.last_servo_response = response
        else:
            self.last_servo_response = "DRY-RUN"
        self.commanded_pose = result.pose
        self.last_command_time = now
        return result

    def hold(self) -> None:
        self.send_pose(self.commanded_pose)

    def close(self) -> None:
        if self.armed:
            self.set_armed(False)
        elif self.robot is not None:
            try:
                self.robot.deactivate_servo_interface()
            except Exception:
                pass
            try:
                self.robot.stop()
            except Exception:
                pass


def draw_hand(frame: Any, landmarks: Sequence[Any]) -> None:
    height, width = frame.shape[:2]
    points = [
        (int(landmark.x * width), int(landmark.y * height)) for landmark in landmarks
    ]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 2)
    for index, point in enumerate(points):
        color = (255, 0, 255) if index == 4 else (0, 255, 0) if index == 8 else (0, 255, 255)
        cv2.circle(frame, point, 5, color, -1)


def open_laptop_camera(args: argparse.Namespace) -> Any:
    camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        raise RuntimeError(
            f"Laptop-Kamera Index {args.camera_index} konnte nicht geoeffnet werden. "
            "Mit --camera-index 1 oder 2 testen und Kamera-Berechtigungen pruefen."
        )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def put_status(frame: Any, lines: Sequence[tuple[str, tuple[int, int, int]]]) -> None:
    for index, (message, color) in enumerate(lines):
        cv2.putText(
            frame,
            message,
            (20, 34 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            color,
            2,
            cv2.LINE_AA,
        )


def validate_args(args: argparse.Namespace) -> None:
    axes = (args.horizontal_axis, args.vertical_axis, args.depth_axis)
    if len(set(axes)) != 3:
        raise SystemExit("horizontal-, vertical- und depth-axis muessen verschieden sein.")
    if args.execute and args.no_gripper:
        print("Hinweis: --no-gripper aktiv; nur LARA wird gesteuert.")
    for name in ("horizontal_sign", "vertical_sign", "depth_sign"):
        if getattr(args, name) not in (-1.0, 1.0):
            raise SystemExit(f"--{name.replace('_', '-')} muss -1 oder 1 sein.")
    for name in (
        "horizontal_scale", "vertical_scale", "depth_scale",
        "max_horizontal", "max_vertical", "max_depth",
        "workspace_x", "workspace_y", "workspace_z",
        "max_speed", "max_tcp_step", "update_hz", "hand_loss_timeout",
        "pinch_threshold", "rotation_pinch_threshold", "rotation_scale",
        "roll_scale", "max_rotation_deg", "max_angular_speed",
        "max_angular_step_deg", "open_width",
    ):
        if getattr(args, name) <= 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} muss groesser als 0 sein.")
    if args.closed_width < 0.0:
        raise SystemExit("--closed-width darf nicht negativ sein.")
    if args.closed_width >= args.open_width:
        raise SystemExit("--closed-width muss kleiner als --open-width sein.")
    for name in ("hand_smoothing", "target_smoothing", "orientation_smoothing"):
        if not 0.0 < getattr(args, name) <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} muss im Bereich (0, 1] liegen.")
    if args.gesture_frames < 1:
        raise SystemExit("--gesture-frames muss mindestens 1 sein.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Laptop-Kamera Handsteuerung fuer NEURA LARA + Contactile"
    )
    parser.add_argument("--execute", action="store_true", help="Echte Hardware freigeben; Steuerung startet automatisch")
    parser.add_argument("--robot-ip", default="192.168.2.13")
    parser.add_argument("--camera-index", type=int, default=0, help="0 ist normalerweise die eingebaute Laptop-Kamera")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("hand_landmarker.task"))
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--gripper-port", default="auto", help="auto, ACM0 oder /dev/ttyACM0")
    parser.add_argument("--open-width", type=float, default=160.0, help="Contactile ganz offen in mm")
    parser.add_argument("--closed-width", type=float, default=0.0, help="Contactile ganz geschlossen in mm")
    parser.add_argument("--gripper-ready-timeout", type=float, default=5.0)
    parser.add_argument("--release-wait", type=float, default=0.15)
    parser.add_argument("--tactile-stream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pinch-threshold", type=float, default=0.75)
    parser.add_argument("--rotation-pinch-threshold", type=float, default=0.62, help="Daumen/Mittelfinger-Pinch fuer TCP-Rotation")
    parser.add_argument("--rotation-scale", type=float, default=2.2, help="Pitch/Yaw Radiant pro normierter Bildbewegung")
    parser.add_argument("--roll-scale", type=float, default=1.0, help="Verstaerkung fuer sichtbare Handdrehung")
    parser.add_argument("--max-rotation-deg", type=float, default=60.0, help="maximale Rotation pro Rotations-Pinch")
    parser.add_argument("--orientation-smoothing", type=float, default=0.25)
    parser.add_argument("--max-angular-speed", type=float, default=0.50, help="maximale TCP-Winkelgeschwindigkeit rad/s")
    parser.add_argument("--max-angular-step-deg", type=float, default=2.0, help="maximaler Rotationsschritt pro Update")
    parser.add_argument("--gesture-frames", type=int, default=8)
    parser.add_argument("--hand-smoothing", type=float, default=0.45)
    parser.add_argument("--target-smoothing", type=float, default=0.28)
    parser.add_argument("--dead-zone", type=float, default=0.002)
    parser.add_argument("--update-hz", type=float, default=20.0)
    parser.add_argument("--hand-loss-timeout", type=float, default=0.7)
    parser.add_argument("--horizontal-axis", choices=AXIS_INDEX, default="x", help="Kamera rechts/links -> LARA-Achse")
    parser.add_argument("--vertical-axis", choices=AXIS_INDEX, default="z", help="Kamera hoch/runter -> LARA-Achse")
    parser.add_argument("--depth-axis", choices=AXIS_INDEX, default="y", help="Handgroesse -> LARA-Achse")
    parser.add_argument("--horizontal-sign", type=float, default=1.0)
    parser.add_argument("--vertical-sign", type=float, default=1.0)
    parser.add_argument("--depth-sign", type=float, default=1.0)
    parser.add_argument("--horizontal-scale", type=float, default=0.90, help="m pro normierter Bildbreite")
    parser.add_argument("--vertical-scale", type=float, default=0.70, help="m pro normierter Bildhoehe")
    parser.add_argument("--depth-scale", type=float, default=0.50, help="m pro relativer Handgroessenaenderung")
    parser.add_argument("--max-horizontal", type=float, default=0.45, help="m ab Pinch-Start")
    parser.add_argument("--max-vertical", type=float, default=0.40, help="m ab Pinch-Start")
    parser.add_argument("--max-depth", type=float, default=0.40, help="m ab Pinch-Start")
    parser.add_argument(
        "--enable-depth",
        dest="enable_depth",
        action="store_true",
        help="Monokulare Tiefenbewegung ueber die Handgroesse aktivieren (Standard)",
    )
    parser.add_argument(
        "--disable-depth",
        dest="enable_depth",
        action="store_false",
        help="Vor-/Zurueckbewegung ueber die Handgroesse deaktivieren",
    )
    parser.set_defaults(enable_depth=True)
    parser.add_argument("--workspace-x", type=float, default=0.45, help="sicherer Radius um Start-TCP in m")
    parser.add_argument("--workspace-y", type=float, default=0.40)
    parser.add_argument("--workspace-z", type=float, default=0.40)
    parser.add_argument("--max-speed", type=float, default=0.05, help="maximale TCP-Geschwindigkeit m/s")
    parser.add_argument("--max-tcp-step", type=float, default=0.004, help="maximaler TCP-Schritt pro Update in m")
    parser.add_argument("--servo-gain", type=float, default=2.0)
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def run(args: argparse.Namespace) -> None:
    if cv2 is None or mp is None:
        raise RuntimeError("Fehlende Kamerapakete. Installieren: pip install opencv-python mediapipe")
    if not args.model.is_file():
        raise FileNotFoundError(f"MediaPipe-Modell fehlt: {args.model}")

    camera = open_laptop_camera(args)
    lara = LaraController(args)
    gripper = ContactileController(args)
    lara.connect()
    if args.execute and not args.no_gripper:
        gripper.connect()

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.60,
        min_hand_presence_confidence=0.60,
        min_tracking_confidence=0.60,
    )

    filtered_palm = None
    filtered_hand_size = None
    control_previous = None
    hand_anchor = None
    hand_size_anchor = None
    roll_anchor = None
    pose_anchor = None
    gesture_candidate = None
    gesture_frames = 0
    last_hand_seen = time.monotonic()
    last_update = 0.0
    last_gesture = "-"
    workspace_limited = False
    start = time.perf_counter()
    timestamp_previous = -1

    print("\nDaumen/Zeigefinger = XYZ | Daumen/Mittelfinger = Rotation | offene Hand/Faust = Gripper | Q/Esc = Ende")
    if args.execute:
        print("Hardwaremodus startet automatisch. E-Stop erreichbar halten.")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            lara.set_armed(True)
            while True:
                ok, frame = camera.read()
                if not ok:
                    if lara.armed:
                        lara.set_armed(False)
                    raise RuntimeError("Laptop-Kamera liefert keine Bilder mehr; LARA wurde gestoppt.")

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = int((time.perf_counter() - start) * 1000)
                timestamp = max(timestamp, timestamp_previous + 1)
                timestamp_previous = timestamp
                result = landmarker.detect_for_video(image, timestamp)

                now = time.monotonic()
                hand_status = "NO HAND"
                control_mode = None

                if result.hand_landmarks:
                    last_hand_seen = now
                    if not lara.armed:
                        print("Hand wieder erkannt: Steuerung wird automatisch aktiviert.")
                        lara.set_armed(True)
                        control_previous = None
                        hand_anchor = None
                        pose_anchor = None
                    landmarks = result.hand_landmarks[0]
                    draw_hand(frame, landmarks)
                    raw_palm = palm_center(landmarks)
                    if filtered_palm is None:
                        filtered_palm = raw_palm
                    else:
                        alpha = args.hand_smoothing
                        filtered_palm = (
                            alpha * raw_palm[0] + (1.0 - alpha) * filtered_palm[0],
                            alpha * raw_palm[1] + (1.0 - alpha) * filtered_palm[1],
                        )

                    hand_size = distance_2d(landmarks[5], landmarks[17])
                    if filtered_hand_size is None:
                        filtered_hand_size = hand_size
                    else:
                        filtered_hand_size = 0.15 * hand_size + 0.85 * filtered_hand_size

                    pinch_ratio = distance_2d(landmarks[4], landmarks[8]) / max(
                        distance_2d(landmarks[5], landmarks[8]), 1e-4
                    )
                    rotation_pinch_ratio = distance_2d(landmarks[4], landmarks[12]) / max(
                        distance_2d(landmarks[9], landmarks[12]), 1e-4
                    )
                    if pinch_ratio < args.pinch_threshold:
                        control_mode = "MOVE XYZ"
                    elif rotation_pinch_ratio < args.rotation_pinch_threshold:
                        control_mode = "ROTATE TCP"
                    hand_status = control_mode or "HAND READY"

                    gesture = detect_gripper_gesture(landmarks, control_mode is not None)
                    if gesture is None:
                        gesture_candidate = None
                        gesture_frames = 0
                    elif gesture == gesture_candidate:
                        gesture_frames += 1
                    else:
                        gesture_candidate = gesture
                        gesture_frames = 1
                    if gesture_frames >= args.gesture_frames and gesture != last_gesture:
                        last_gesture = gesture
                        if lara.armed and not args.no_gripper:
                            gripper.command(gesture)

                    if control_mode is not None and control_mode != control_previous:
                        hand_anchor = filtered_palm
                        hand_size_anchor = filtered_hand_size
                        roll_anchor = palm_roll_angle(landmarks)
                        pose_anchor = lara.current_pose()

                    if control_mode == "MOVE XYZ" and hand_anchor is not None and pose_anchor is not None:
                        dx = filtered_palm[0] - hand_anchor[0]
                        dy = filtered_palm[1] - hand_anchor[1]
                        if abs(dx) < args.dead_zone:
                            dx = 0.0
                        if abs(dy) < args.dead_zone:
                            dy = 0.0
                        depth = 0.0
                        if args.enable_depth and hand_size_anchor:
                            depth = filtered_hand_size / max(hand_size_anchor, 1e-4) - 1.0
                        position = map_hand_delta(pose_anchor[:3], dx, dy, depth, args)
                        desired_pose = position + pose_anchor[3:7]
                        if lara.armed and now - last_update >= 1.0 / args.update_hz:
                            motion = lara.send_pose(desired_pose)
                            workspace_limited = motion.workspace_limited
                            last_update = now
                    elif (
                        control_mode == "ROTATE TCP"
                        and hand_anchor is not None
                        and roll_anchor is not None
                        and pose_anchor is not None
                    ):
                        dx = filtered_palm[0] - hand_anchor[0]
                        dy = filtered_palm[1] - hand_anchor[1]
                        if abs(dx) < args.dead_zone:
                            dx = 0.0
                        if abs(dy) < args.dead_zone:
                            dy = 0.0
                        roll_delta = wrap_angle(palm_roll_angle(landmarks) - roll_anchor)
                        quaternion = map_hand_rotation(
                            pose_anchor[3:7], dx, dy, roll_delta, args
                        )
                        desired_pose = pose_anchor[:3] + quaternion
                        if lara.armed and now - last_update >= 1.0 / args.update_hz:
                            motion = lara.send_pose(desired_pose)
                            workspace_limited = motion.workspace_limited
                            last_update = now
                    elif lara.armed and now - last_update >= 1.0 / args.update_hz:
                        lara.hold()
                        last_update = now

                    control_previous = control_mode
                else:
                    control_previous = None
                    hand_anchor = None
                    roll_anchor = None
                    pose_anchor = None
                    gesture_candidate = None
                    gesture_frames = 0
                    if lara.armed and now - last_hand_seen > args.hand_loss_timeout:
                        print("Hand verloren: LARA wird automatisch gestoppt.")
                        lara.set_armed(False)
                    elif lara.armed and now - last_update >= 1.0 / args.update_hz:
                        lara.hold()
                        last_update = now

                mode = "AKTIV" if lara.armed else "WARTE AUF HAND"
                mode_color = (0, 220, 0) if lara.armed else (0, 0, 255)
                put_status(
                    frame,
                    (
                        (f"{mode} | {'HARDWARE' if args.execute else 'CAMERA TEST'}", mode_color),
                        (hand_status, (0, 220, 0) if control_mode is not None else (0, 200, 255)),
                        (f"GRIPPER: {last_gesture}", (255, 200, 0)),
                        ("WORKSPACE LIMIT" if workspace_limited else "Q: quit", (0, 0, 255) if workspace_limited else (220, 220, 220)),
                    ),
                )
                cv2.imshow("LARA + Contactile Mimic", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        lara.close()
        camera.release()
        cv2.destroyAllWindows()


def main(argv: Sequence[str] | None = None) -> int:
    install_cleanup_signal_handlers()
    try:
        run(parse_args(argv))
    except KeyboardInterrupt:
        print("\nAbgebrochen.")
        return 130
    except Exception as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
