#!/usr/bin/env python3
"""Direct two-hand controller for NEURA LARA and a Contactile gripper.

Right hand: motion clutch, gripper and waypoint gestures.
Left hand: Cartesian axis/mode and individual joint selection.
Hardware is only addressed when --execute is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import lara_contactile_mimic as core

try:
    import ruckig
except ImportError:  # Reported only when JOINT mode is actually used.
    ruckig = None


cv2 = core.cv2
mp = core.mp
CONTROL_MODES = ("CARTESIAN", "JOINT")
AXIS_MODES = ("PAUSE", "X", "Y", "Z", "FREE")


def get_hands_by_label(result: Any, swap_hand_labels: bool = True) -> dict[str, Any]:
    """Map MediaPipe handedness labels to landmarks in the mirrored preview."""
    hands: dict[str, Any] = {}
    for landmarks, handedness in zip(result.hand_landmarks or (), result.handedness or ()):
        if not handedness:
            continue
        label = handedness[0].category_name
        if swap_hand_labels:
            label = {"Left": "Right", "Right": "Left"}.get(label, label)
        hands[label] = landmarks
    return hands


def count_extended_fingers(landmarks: Sequence[Any]) -> int:
    center = core.palm_center(landmarks)
    return sum(
        core.finger_extended(landmarks, tip, pip, center)
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
    )


def detect_control_mode_gesture(
    landmarks: Sequence[Any], thumb_mode_margin: float = 0.04
) -> str | None:
    """Return CARTESIAN for thumb-up and JOINT for thumb-down."""
    if count_extended_fingers(landmarks) != 0:
        return None
    center = core.palm_center(landmarks)
    tip, mcp = landmarks[4], landmarks[2]
    tip_distance = math.hypot(tip.x - center[0], tip.y - center[1])
    mcp_distance = math.hypot(mcp.x - center[0], mcp.y - center[1])
    if tip_distance <= mcp_distance * 1.25:
        return None
    if tip.y < mcp.y - thumb_mode_margin:
        return "CARTESIAN"
    if tip.y > mcp.y + thumb_mode_margin:
        return "JOINT"
    return None


def detect_axis_mode(landmarks: Sequence[Any]) -> str:
    return {0: "PAUSE", 1: "X", 2: "Y", 3: "Z", 4: "FREE"}[
        count_extended_fingers(landmarks)
    ]


def is_thumb_extended_for_joint(landmarks: Sequence[Any]) -> bool:
    palm_width = core.distance_2d(landmarks[5], landmarks[17])
    return palm_width >= 0.001 and (
        core.distance_2d(landmarks[4], landmarks[5]) / palm_width
    ) > 0.75


def detect_joint_selection_gesture(landmarks: Sequence[Any]) -> int | None:
    """Map left-hand finger patterns to LARA joints J1 through J6."""
    center = core.palm_center(landmarks)
    index, middle, ring, little = (
        core.finger_extended(landmarks, tip, pip, center)
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
    )
    if not any((index, middle, ring, little)):
        return 6
    if index and not any((middle, ring, little)):
        return 1
    if index and middle and not any((ring, little)):
        return 2
    if index and middle and ring and not little:
        return 3
    if index and middle and ring and little:
        return 5 if is_thumb_extended_for_joint(landmarks) else 4
    return None


def apply_axis_lock(
    anchor_position: Sequence[float], desired_position: Sequence[float], axis_mode: str
) -> list[float]:
    """Keep only the selected LARA base-frame displacement."""
    anchor = [float(value) for value in anchor_position[:3]]
    desired = [float(value) for value in desired_position[:3]]
    if axis_mode == "FREE":
        return desired
    if axis_mode != "PAUSE":
        index = core.AXIS_INDEX[axis_mode.lower()]
        anchor[index] = desired[index]
    return anchor


def map_joint_target(
    anchor_joints: Sequence[float],
    selected_joint: int,
    horizontal_delta: float,
    joint_scale_deg: float,
    max_from_anchor_deg: float,
) -> list[float]:
    """Map horizontal hand displacement to one joint in LARA radians."""
    if selected_joint < 1 or selected_joint > len(anchor_joints):
        raise ValueError(f"Ungueltiges LARA-Gelenk: J{selected_joint}")
    limit = math.radians(max_from_anchor_deg)
    change = core.clamp(math.radians(horizontal_delta * joint_scale_deg), -limit, limit)
    target = [float(value) for value in anchor_joints]
    target[selected_joint - 1] += change
    return target


def detect_save_waypoint_gesture(
    landmarks: Sequence[Any], pinch_active: bool, separation_ratio: float = 0.20
) -> bool:
    if pinch_active:
        return False
    center = core.palm_center(landmarks)
    index, middle, ring, little = (
        core.finger_extended(landmarks, tip, pip, center)
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
    )
    palm_width = core.distance_2d(landmarks[5], landmarks[17])
    separated = core.distance_2d(landmarks[8], landmarks[12]) > palm_width * separation_ratio
    return index and middle and not ring and not little and separated


class DirectLaraController(core.LaraController):
    """Extend the shared direct Cartesian controller with bounded joint servoing."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.commanded_joints = [0.0] * 6
        self.commanded_joint_velocity = [0.0] * 6
        self.commanded_joint_acceleration = [0.0] * 6
        self.joint_workspace_center = [0.0] * 6
        self._joint_lock = threading.RLock()
        self._joint_stream_stop = threading.Event()
        self._joint_stream_thread: threading.Thread | None = None
        self._joint_target = list(self.commanded_joints)
        self._joint_target_time = 0.0
        self._joint_target_revision = 0
        self._joint_motion_enabled = False
        self._joint_stream_error: Exception | None = None
        self._joint_stream_window_start = time.monotonic()
        self._joint_stream_calls = 0
        self.joint_stream_rate = 0.0

    def _read_pose_as_reference(self) -> None:
        # Keep the activation sequence identical to the working one-hand
        # controller: init_program -> TCP reference -> activate servo. Some
        # controller versions reject activation when another API request is
        # inserted between the TCP read and activate_servo_interface().
        super()._read_pose_as_reference()

    def _read_joints_as_reference(self) -> None:
        with self._joint_lock:
            if self.robot is not None:
                if not hasattr(self.robot, "get_current_joint_angles"):
                    raise RuntimeError("Die neurapy-Version bietet get_current_joint_angles() nicht an.")
                joints = self.robot.get_current_joint_angles()
                if not joints or len(joints) < 6:
                    raise RuntimeError(f"Ungueltige LARA-Gelenkpositionen: {joints!r}")
                self.commanded_joints = [float(value) for value in joints]
            self.commanded_joint_velocity = [0.0] * len(self.commanded_joints)
            self.commanded_joint_acceleration = [0.0] * len(self.commanded_joints)
            self.joint_workspace_center = list(self.commanded_joints)
            self._joint_target = list(self.commanded_joints)
            self._joint_motion_enabled = False

    def set_armed(self, armed: bool) -> None:
        if not armed:
            self.stop_joint_stream()
        super().set_armed(armed)
        if armed:
            # Joint state is needed only for JOINT mode and is deliberately
            # fetched after the position servo interface is active.
            try:
                self._read_joints_as_reference()
            except Exception:
                super().set_armed(False)
                raise

    def current_joints(self) -> list[float]:
        with self._joint_lock:
            if self.robot is not None:
                joints = self.robot.get_current_joint_angles()
                if joints and len(joints) >= 6:
                    self.commanded_joints = [float(value) for value in joints]
                    self.commanded_joint_velocity = [0.0] * len(self.commanded_joints)
                    self.commanded_joint_acceleration = [0.0] * len(self.commanded_joints)
            return list(self.commanded_joints)

    def send_joints(
        self, desired_joints: Sequence[float], immediate_stop: bool = False
    ) -> list[float]:
        with self._joint_lock:
            return self._send_joints_locked(desired_joints, immediate_stop)

    def _send_joints_locked(
        self, desired_joints: Sequence[float], immediate_stop: bool = False
    ) -> list[float]:
        if len(desired_joints) != len(self.commanded_joints):
            raise ValueError(
                f"LARA meldet {len(self.commanded_joints)} Gelenke, Ziel hat {len(desired_joints)}."
            )
        now = time.monotonic()
        dt = max(0.001, now - self.last_command_time)
        workspace_limit = math.radians(self.args.joint_workspace_deg)
        max_step = math.radians(self.args.max_joint_step_deg)
        max_velocity_change = self.args.max_joint_acceleration * dt
        target = []
        velocity = []
        for current, previous_velocity, desired, center in zip(
            self.commanded_joints,
            self.commanded_joint_velocity,
            desired_joints,
            self.joint_workspace_center,
        ):
            bounded = core.clamp(float(desired), center - workspace_limit, center + workspace_limit)
            error = bounded - current
            if immediate_stop:
                next_velocity = 0.0
                step = 0.0
            else:
                requested_velocity = core.clamp(
                    error * self.args.joint_smoothing / dt,
                    -self.args.max_joint_speed,
                    self.args.max_joint_speed,
                )
                next_velocity = previous_velocity + core.clamp(
                    requested_velocity - previous_velocity,
                    -max_velocity_change,
                    max_velocity_change,
                )
                step = core.clamp(next_velocity * dt, -max_step, max_step)
                if abs(step) > abs(error):
                    step = error
                next_velocity = step / dt
            target.append(current + step)
            velocity.append(next_velocity)
        acceleration = [0.0] * len(target)
        return self._send_joint_state_locked(target, velocity, acceleration)

    def _send_joint_state_locked(
        self,
        target: Sequence[float],
        velocity: Sequence[float],
        acceleration: Sequence[float],
    ) -> list[float]:
        target = [float(value) for value in target]
        velocity = [float(value) for value in velocity]
        acceleration = [float(value) for value in acceleration]
        if self.robot is not None:
            if not hasattr(self.robot, "servo_j"):
                raise RuntimeError("Die neurapy-Version bietet servo_j() nicht an.")
            # LARA server v5.0.8 exposes servo_j(position, velocity, acceleration)
            # without the ServoX-only gain argument.
            response = self.robot.servo_j(target, velocity, acceleration)
            if isinstance(response, (int, float)) and response >= 3:
                raise RuntimeError(f"LARA servo_j Fehlercode: {response}")
            self.last_servo_response = response
        else:
            self.last_servo_response = "DRY-RUN"
        self.commanded_joints = target
        self.commanded_joint_velocity = velocity
        self.commanded_joint_acceleration = acceleration
        self.last_command_time = time.monotonic()
        return list(target)

    def _joint_stream_loop(self) -> None:
        period = 1.0 / self.args.joint_servo_hz
        next_tick = time.monotonic()
        planned_revision = -1
        trajectory = None
        trajectory_start = next_tick
        try:
            while not self._joint_stream_stop.is_set():
                now = time.monotonic()
                with self._joint_lock:
                    target_is_fresh = (
                        self._joint_motion_enabled
                        and now - self._joint_target_time
                        <= self.args.joint_command_timeout
                    )
                    if not target_is_fresh:
                        trajectory = None
                        planned_revision = -1
                        zeros = [0.0] * len(self.commanded_joints)
                        self._send_joint_state_locked(
                            self.commanded_joints, zeros, zeros
                        )
                    else:
                        if self._joint_target_revision != planned_revision:
                            dof = len(self.commanded_joints)
                            inp = ruckig.InputParameter(dof)
                            inp.current_position = list(self.commanded_joints)
                            inp.current_velocity = list(self.commanded_joint_velocity)
                            inp.current_acceleration = list(
                                self.commanded_joint_acceleration
                            )
                            workspace_limit = math.radians(
                                self.args.joint_workspace_deg
                            )
                            inp.target_position = [
                                core.clamp(
                                    float(target),
                                    center - workspace_limit,
                                    center + workspace_limit,
                                )
                                for target, center in zip(
                                    self._joint_target,
                                    self.joint_workspace_center,
                                )
                            ]
                            inp.target_velocity = [0.0] * dof
                            inp.target_acceleration = [0.0] * dof
                            inp.max_velocity = [self.args.max_joint_speed] * dof
                            inp.max_acceleration = [
                                self.args.max_joint_acceleration
                            ] * dof
                            inp.max_jerk = [self.args.max_joint_jerk] * dof
                            trajectory = ruckig.Trajectory(dof)
                            result = ruckig.Ruckig(dof).calculate(inp, trajectory)
                            if result not in (
                                ruckig.Result.Working, ruckig.Result.Finished
                            ):
                                raise RuntimeError(
                                    f"Ruckig konnte Joint-Trajektorie nicht berechnen: {result}"
                                )
                            trajectory_start = now
                            planned_revision = self._joint_target_revision
                        position, velocity, acceleration = trajectory.at_time(
                            max(0.0, now - trajectory_start)
                        )
                        self._send_joint_state_locked(
                            position, velocity, acceleration
                        )
                    self._joint_stream_calls += 1
                    rate_elapsed = now - self._joint_stream_window_start
                    if rate_elapsed >= 1.0:
                        self.joint_stream_rate = self._joint_stream_calls / rate_elapsed
                        self._joint_stream_calls = 0
                        self._joint_stream_window_start = now
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay <= 0.0:
                    next_tick = time.monotonic()
                    # Always yield briefly when the JSON-RPC roundtrip is
                    # slower than the requested cycle. Otherwise this worker
                    # starves camera processing without increasing servo rate.
                    self._joint_stream_stop.wait(min(period, 0.001))
                    continue
                self._joint_stream_stop.wait(delay)
        except Exception as exc:
            with self._joint_lock:
                self._joint_stream_error = exc
                self._joint_motion_enabled = False
            self._joint_stream_stop.set()

    def _start_joint_stream(self) -> None:
        if ruckig is None:
            raise RuntimeError(
                "ruckig fehlt. In .venv310 mit 'pip install ruckig==0.19.4' installieren."
            )
        self.check_joint_stream()
        if self._joint_stream_thread is not None and self._joint_stream_thread.is_alive():
            return
        self._joint_stream_stop.clear()
        self._joint_stream_window_start = time.monotonic()
        self._joint_stream_calls = 0
        self.joint_stream_rate = 0.0
        self._joint_stream_thread = threading.Thread(
            target=self._joint_stream_loop,
            name="lara-servo-j",
            daemon=True,
        )
        self._joint_stream_thread.start()

    def set_joint_target(self, desired_joints: Sequence[float]) -> None:
        if len(desired_joints) != len(self.commanded_joints):
            raise ValueError(
                f"LARA meldet {len(self.commanded_joints)} Gelenke, "
                f"Ziel hat {len(desired_joints)}."
            )
        self._start_joint_stream()
        with self._joint_lock:
            self._joint_target = [float(value) for value in desired_joints]
            self._joint_target_time = time.monotonic()
            self._joint_target_revision += 1
            self._joint_motion_enabled = True

    def hold_joint_motion(self) -> None:
        with self._joint_lock:
            self._joint_target = list(self.commanded_joints)
            self._joint_target_time = time.monotonic()
            self._joint_motion_enabled = False

    def check_joint_stream(self) -> None:
        with self._joint_lock:
            error = self._joint_stream_error
        if error is not None:
            raise RuntimeError(f"LARA servo_j Hintergrundtakt fehlgeschlagen: {error}") from error

    def stop_joint_stream(self, require_stopped: bool = False) -> None:
        thread = self._joint_stream_thread
        if thread is None:
            return
        self.hold_joint_motion()
        self._joint_stream_stop.set()
        if thread is not threading.current_thread():
            thread.join(timeout=1.0)
        if thread.is_alive():
            message = "LARA servo_j Hintergrundtakt reagiert nicht auf Stop."
            if require_stopped:
                raise RuntimeError(message)
            print(f"WARNUNG: {message}")
        self._joint_stream_thread = None

    def hold_mode(self, control_mode: str) -> None:
        if control_mode == "JOINT":
            if (
                self._joint_stream_thread is not None
                and self._joint_stream_thread.is_alive()
            ):
                self.hold_joint_motion()
            else:
                self.send_joints(self.commanded_joints, immediate_stop=True)
        else:
            self.stop_joint_stream(require_stopped=True)
            self.hold()


def save_waypoint(
    waypoints: list[dict[str, Any]],
    lara: DirectLaraController,
    gripper_state: str,
    output_path: Path,
    min_distance: float,
) -> bool:
    pose = lara.current_pose()
    if waypoints:
        distance = math.sqrt(
            sum((pose[index] - waypoints[-1]["pose"][index]) ** 2 for index in range(3))
        )
        if distance < min_distance:
            return False
    waypoints.append(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "pose": pose,
            "joints": lara.current_joints(),
            "gripper": gripper_state,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(waypoints, indent=2) + "\n", encoding="utf-8")
    print(f"Wegpunkt {len(waypoints)} gespeichert: {output_path}")
    return True


def draw_hand(frame: Any, landmarks: Sequence[Any], label: str, color: tuple[int, int, int]) -> None:
    core.draw_hand(frame, landmarks)
    height, width = frame.shape[:2]
    point = (
        max(10, int(landmarks[0].x * width) - 35),
        min(height - 10, max(25, int(landmarks[0].y * height) + 30)),
    )
    cv2.putText(frame, label, point, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def validate_args(args: argparse.Namespace) -> None:
    core.validate_args(args)
    if args.control_hand == args.axis_hand:
        raise SystemExit("--control-hand und --axis-hand muessen verschieden sein.")
    for name in (
        "axis_debounce_frames", "joint_select_debounce_frames", "joint_scale_deg",
        "joint_max_from_anchor_deg", "joint_workspace_deg", "joint_smoothing",
        "max_joint_speed", "max_joint_acceleration", "max_joint_jerk",
        "max_joint_step_deg",
        "joint_servo_hz", "joint_command_timeout", "mode_switch_hold",
        "waypoint_hold", "waypoint_min_distance",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} muss groesser als 0 sein.")
    if args.joint_smoothing > 1.0:
        raise SystemExit("--joint-smoothing muss im Bereich (0, 1] liegen.")
    if args.joint_servo_hz > 250.0:
        raise SystemExit("--joint-servo-hz darf hoechstens 250 sein.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zwei-Hand-Steuerung fuer NEURA LARA + Contactile (direkte API)"
    )
    parser.add_argument("--execute", action="store_true", help="Echte Hardware freigeben")
    parser.add_argument("--robot-ip", default="192.168.2.13")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("hand_landmarker.task"))
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--gripper-port", default="auto")
    parser.add_argument("--open-width", type=float, default=160.0)
    parser.add_argument("--closed-width", type=float, default=0.0)
    parser.add_argument("--gripper-ready-timeout", type=float, default=5.0)
    parser.add_argument("--release-wait", type=float, default=0.15)
    parser.add_argument("--tactile-stream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pinch-threshold", type=float, default=0.75)
    parser.add_argument("--gesture-frames", type=int, default=8)
    parser.add_argument("--hand-smoothing", type=float, default=0.25)
    parser.add_argument("--target-smoothing", type=float, default=0.28)
    parser.add_argument("--dead-zone", type=float, default=0.002)
    parser.add_argument("--update-hz", type=float, default=30.0)
    parser.add_argument("--hand-loss-timeout", type=float, default=0.7)
    parser.add_argument("--horizontal-axis", choices=core.AXIS_INDEX, default="x")
    parser.add_argument("--vertical-axis", choices=core.AXIS_INDEX, default="z")
    parser.add_argument("--depth-axis", choices=core.AXIS_INDEX, default="y")
    parser.add_argument("--horizontal-sign", type=float, default=1.0)
    parser.add_argument("--vertical-sign", type=float, default=1.0)
    parser.add_argument("--depth-sign", type=float, default=1.0)
    parser.add_argument("--horizontal-scale", type=float, default=0.90)
    parser.add_argument("--vertical-scale", type=float, default=0.70)
    parser.add_argument("--depth-scale", type=float, default=0.50)
    parser.add_argument("--max-horizontal", type=float, default=0.45)
    parser.add_argument("--max-vertical", type=float, default=0.40)
    parser.add_argument("--max-depth", type=float, default=0.40)
    parser.add_argument("--enable-depth", dest="enable_depth", action="store_true")
    parser.add_argument("--disable-depth", dest="enable_depth", action="store_false")
    parser.set_defaults(enable_depth=True)
    parser.add_argument("--workspace-x", type=float, default=0.45)
    parser.add_argument("--workspace-y", type=float, default=0.40)
    parser.add_argument("--workspace-z", type=float, default=0.40)
    parser.add_argument("--max-speed", type=float, default=0.05)
    parser.add_argument("--max-tcp-step", type=float, default=0.004)
    parser.add_argument("--servo-gain", type=float, default=2.0)
    parser.add_argument("--orientation-smoothing", type=float, default=0.25)
    parser.add_argument("--max-angular-speed", type=float, default=0.50)
    parser.add_argument("--max-angular-step-deg", type=float, default=2.0)
    parser.set_defaults(
        rotation_pinch_threshold=0.62, rotation_scale=2.2,
        roll_scale=1.0, max_rotation_deg=60.0,
    )
    parser.add_argument("--control-hand", choices=("Left", "Right"), default="Right")
    parser.add_argument("--axis-hand", choices=("Left", "Right"), default="Left")
    parser.add_argument("--swap-hand-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-axis-hand",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Linke Auswahlhand permanent verlangen. Standard: Die linke Hand waehlt "
            "Achse/Modus und darf danach das Bild verlassen."
        ),
    )
    parser.add_argument("--axis-mode-default", choices=AXIS_MODES, default="PAUSE")
    parser.add_argument("--axis-debounce-frames", type=int, default=6)
    parser.add_argument("--mode-default", choices=CONTROL_MODES, default="CARTESIAN")
    parser.add_argument("--mode-switch-hold", type=float, default=0.50)
    parser.add_argument("--thumb-mode-margin", type=float, default=0.04)
    parser.add_argument("--joint-select-debounce-frames", type=int, default=6)
    parser.add_argument("--joint-scale-deg", type=float, default=180.0)
    parser.add_argument("--joint-max-from-anchor-deg", type=float, default=45.0)
    parser.add_argument("--joint-workspace-deg", type=float, default=60.0)
    parser.add_argument("--joint-smoothing", type=float, default=0.30)
    parser.add_argument(
        "--max-joint-speed", type=float, default=0.03,
        help="Maximale Gelenkgeschwindigkeit in rad/s (Standard: 0.03 = ca. 1.7 Grad/s)",
    )
    parser.add_argument(
        "--max-joint-acceleration", type=float, default=0.12,
        help="Interne maximale Gelenkbeschleunigung in rad/s^2 (Standard: 0.12)",
    )
    parser.add_argument(
        "--max-joint-jerk", type=float, default=0.50,
        help="Maximaler Gelenkruck fuer Ruckig in rad/s^3 (Standard: 0.50)",
    )
    parser.add_argument(
        "--max-joint-step-deg", type=float, default=0.5,
        help="Zusaetzliche Begrenzung pro Servo-Befehl in Grad (Standard: 0.5)",
    )
    parser.add_argument(
        "--joint-servo-hz", type=float, default=100.0,
        help="Ruckig-servo_j-Takt in Hz (Standard: 100; API-Limit ca. 220)",
    )
    parser.add_argument(
        "--joint-command-timeout", type=float, default=0.20,
        help="Ohne frisches Handziel nach dieser Zeit Joint-Bewegung stoppen (Sekunden)",
    )
    parser.add_argument("--waypoint-hold", type=float, default=0.50)
    parser.add_argument("--waypoint-min-distance", type=float, default=0.02, help="m")
    parser.add_argument(
        "--waypoint-file", type=Path,
        default=Path(__file__).with_name("recordings")
        / f"lara_2hand_waypoints_{datetime.now():%Y%m%d_%H%M%S}.json",
    )
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def run(args: argparse.Namespace) -> None:
    if cv2 is None or mp is None:
        raise RuntimeError("Fehlende Kamerapakete. Installieren: pip install opencv-python mediapipe")
    if not args.model.is_file():
        raise FileNotFoundError(f"MediaPipe-Modell fehlt: {args.model}")

    camera = core.open_laptop_camera(args)
    lara = DirectLaraController(args)
    gripper = core.ContactileController(args)
    try:
        lara.connect()
        if args.execute and not args.no_gripper:
            gripper.connect()
    except Exception:
        lara.close()
        camera.release()
        raise

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.60,
        min_hand_presence_confidence=0.60,
        min_tracking_confidence=0.60,
    )

    filtered_palm: tuple[float, float] | None = None
    filtered_hand_size: float | None = None
    hand_anchor: tuple[float, float] | None = None
    hand_size_anchor: float | None = None
    pose_anchor: list[float] | None = None
    joint_anchor: list[float] | None = None
    pinch_previous = False
    axis_mode = args.axis_mode_default
    axis_candidate: str | None = None
    axis_frames = 0
    control_mode = args.mode_default
    selected_joint: int | None = None
    joint_candidate: int | None = None
    joint_frames = 0
    mode_candidate: str | None = None
    mode_since: float | None = None
    rearm_after_mode_change = False
    gripper_candidate: str | None = None
    gripper_frames = 0
    gripper_state = "-"
    waypoint_since: float | None = None
    waypoint_latched = False
    waypoint_status = "READY"
    waypoints: list[dict[str, Any]] = []
    manual_stop = False
    last_required_hands_seen = time.monotonic()
    last_update = 0.0
    last_motion_log = 0.0
    workspace_limited = False
    start = time.perf_counter()
    previous_timestamp = -1

    print("\n2-Hand Mimic: rechts = Bewegung/Gripper, links = Achse/Modus")
    print("Links: Faust PAUSE, 1/2/3/4 Finger X/Y/Z/FREE, Daumen hoch/runter CARTESIAN/JOINT")
    print("Rechts: Pinch bewegen, offene Hand/Faust Gripper, V-Zeichen Wegpunkt")
    print("STOP: Leertaste oder S | Wieder freigeben: A | Ende: Q/Esc")
    if args.execute:
        print("Hardwaremodus startet automatisch. E-Stop erreichbar halten.")

    def reset_motion_anchors() -> None:
        nonlocal hand_anchor, hand_size_anchor, pose_anchor, joint_anchor, pinch_previous
        hand_anchor = None
        hand_size_anchor = None
        pose_anchor = None
        joint_anchor = None
        pinch_previous = False

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            lara.set_armed(True)
            while True:
                lara.check_joint_stream()
                ok, frame = camera.read()
                if not ok:
                    if lara.armed:
                        lara.set_armed(False)
                    raise RuntimeError("Laptop-Kamera liefert keine Bilder mehr; LARA wurde gestoppt.")

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = max(
                    int((time.perf_counter() - start) * 1000), previous_timestamp + 1
                )
                previous_timestamp = timestamp
                result = landmarker.detect_for_video(image, timestamp)
                now = time.monotonic()
                hands = get_hands_by_label(result, args.swap_hand_labels)
                control_hand = hands.get(args.control_hand)
                axis_hand = hands.get(args.axis_hand)
                if control_hand is not None:
                    draw_hand(frame, control_hand, "CONTROL", (0, 255, 0))
                if axis_hand is not None:
                    draw_hand(frame, axis_hand, "AXIS/MODE", (255, 165, 0))

                required_hands_present = control_hand is not None and (
                    axis_hand is not None or not args.require_axis_hand
                )
                mode_gesture_active = False
                if required_hands_present:
                    last_required_hands_seen = now
                    if not lara.armed and not manual_stop:
                        print("Steuerhand wieder erkannt: LARA wird aktiviert.")
                        lara.set_armed(True)
                        reset_motion_anchors()
                elif lara.armed and now - last_required_hands_seen > args.hand_loss_timeout:
                    print("Steuerhand verloren: LARA wird gestoppt.")
                    lara.set_armed(False)
                    reset_motion_anchors()

                # Left hand: mode, axis and joint selection.
                if axis_hand is not None:
                    detected_mode = detect_control_mode_gesture(
                        axis_hand, args.thumb_mode_margin
                    )
                    mode_gesture_active = detected_mode is not None
                    if detected_mode is None:
                        mode_candidate = None
                        mode_since = None
                        if control_mode == "CARTESIAN":
                            detected_axis = detect_axis_mode(axis_hand)
                            if detected_axis == axis_candidate:
                                axis_frames += 1
                            else:
                                axis_candidate, axis_frames = detected_axis, 1
                            if (
                                axis_frames >= args.axis_debounce_frames
                                and detected_axis != axis_mode
                            ):
                                axis_mode = detected_axis
                                print(f"LARA-Achsenmodus: {axis_mode}")
                                reset_motion_anchors()
                        else:
                            detected_joint = detect_joint_selection_gesture(axis_hand)
                            if detected_joint == joint_candidate:
                                joint_frames += 1
                            else:
                                joint_candidate, joint_frames = detected_joint, 1
                            if (
                                detected_joint is not None
                                and joint_frames >= args.joint_select_debounce_frames
                                and detected_joint != selected_joint
                            ):
                                selected_joint = detected_joint
                                print(f"LARA-Gelenk ausgewaehlt: J{selected_joint}")
                                reset_motion_anchors()
                    elif detected_mode != mode_candidate:
                        mode_candidate, mode_since = detected_mode, now
                    elif mode_since is not None and now - mode_since >= args.mode_switch_hold:
                        if control_mode != detected_mode:
                            if detected_mode == "CARTESIAN":
                                lara.stop_joint_stream(require_stopped=True)
                            control_mode = detected_mode
                            rearm_after_mode_change = True
                            reset_motion_anchors()
                            print(f"LARA-Steuermodus: {control_mode}; rechten Pinch einmal loesen.")
                        mode_since = now
                else:
                    axis_candidate, axis_frames = None, 0
                    joint_candidate, joint_frames = None, 0

                # Right hand: clutch, gripper, waypoint and motion.
                pinch_active = False
                pinch_ratio = 0.0
                motion_status = "NO CONTROL HAND"
                if control_hand is not None:
                    raw_palm = core.palm_center(control_hand)
                    if filtered_palm is None:
                        filtered_palm = raw_palm
                    else:
                        alpha = args.hand_smoothing
                        filtered_palm = (
                            alpha * raw_palm[0] + (1.0 - alpha) * filtered_palm[0],
                            alpha * raw_palm[1] + (1.0 - alpha) * filtered_palm[1],
                        )
                    hand_size = core.distance_2d(control_hand[5], control_hand[17])
                    filtered_hand_size = hand_size if filtered_hand_size is None else (
                        0.15 * hand_size + 0.85 * filtered_hand_size
                    )
                    pinch_ratio = core.distance_2d(control_hand[4], control_hand[8]) / max(
                        core.distance_2d(control_hand[5], control_hand[8]), 1e-4
                    )
                    pinch_active = pinch_ratio < args.pinch_threshold
                    if rearm_after_mode_change and not pinch_active:
                        rearm_after_mode_change = False

                    waypoint_gesture = detect_save_waypoint_gesture(
                        control_hand, pinch_active
                    )
                    if waypoint_gesture:
                        if waypoint_since is None:
                            waypoint_since = now
                        waypoint_status = f"HOLD V {now - waypoint_since:.1f}s"
                        if (
                            not waypoint_latched
                            and now - waypoint_since >= args.waypoint_hold
                        ):
                            saved = save_waypoint(
                                waypoints, lara, gripper_state, args.waypoint_file,
                                args.waypoint_min_distance,
                            )
                            waypoint_status = "SAVED" if saved else "TOO CLOSE"
                            waypoint_latched = True
                    else:
                        waypoint_since = None
                        waypoint_latched = False
                        waypoint_status = "READY"

                    gripper_gesture = None if waypoint_gesture else core.detect_gripper_gesture(
                        control_hand, pinch_active
                    )
                    if gripper_gesture is None:
                        gripper_candidate, gripper_frames = None, 0
                    elif gripper_gesture == gripper_candidate:
                        gripper_frames += 1
                    else:
                        gripper_candidate, gripper_frames = gripper_gesture, 1
                    if (
                        gripper_frames >= args.gesture_frames
                        and gripper_gesture != gripper_state
                    ):
                        gripper_state = gripper_gesture
                        if lara.armed and not args.no_gripper:
                            gripper.command(gripper_gesture)

                    movement_allowed = (
                        lara.armed
                        and not manual_stop
                        and required_hands_present
                        and pinch_active
                        and not mode_gesture_active
                        and not rearm_after_mode_change
                        and (control_mode != "CARTESIAN" or axis_mode != "PAUSE")
                        and (control_mode != "JOINT" or selected_joint is not None)
                    )
                    if movement_allowed:
                        if not pinch_previous or hand_anchor is None:
                            hand_anchor = filtered_palm
                            hand_size_anchor = filtered_hand_size
                            if control_mode == "CARTESIAN":
                                pose_anchor = lara.current_pose()
                            else:
                                joint_anchor = lara.current_joints()
                        dx = filtered_palm[0] - hand_anchor[0]
                        dy = filtered_palm[1] - hand_anchor[1]
                        dx = 0.0 if abs(dx) < args.dead_zone else dx
                        dy = 0.0 if abs(dy) < args.dead_zone else dy
                        if control_mode == "CARTESIAN" and pose_anchor is not None:
                            lara.stop_joint_stream(require_stopped=True)
                            depth = 0.0
                            if args.enable_depth and hand_size_anchor:
                                depth = filtered_hand_size / max(hand_size_anchor, 1e-4) - 1.0
                            desired_position = core.map_hand_delta(
                                pose_anchor[:3], dx, dy, depth, args
                            )
                            desired_position = apply_axis_lock(
                                pose_anchor[:3], desired_position, axis_mode
                            )
                            if now - last_update >= 1.0 / args.update_hz:
                                motion = lara.send_pose(desired_position + pose_anchor[3:7])
                                workspace_limited = motion.workspace_limited
                                last_update = now
                                if now - last_motion_log >= 1.0:
                                    delta_mm = [
                                        1000.0 * (motion.pose[index] - pose_anchor[index])
                                        for index in range(3)
                                    ]
                                    print(
                                        "LARA servo_x: "
                                        f"axis={axis_mode} delta_mm="
                                        f"[{delta_mm[0]:+.1f}, {delta_mm[1]:+.1f}, {delta_mm[2]:+.1f}] "
                                        f"response={lara.last_servo_response}"
                                    )
                                    last_motion_log = now
                            motion_status = f"PINCH: {axis_mode}"
                        elif joint_anchor is not None and selected_joint is not None:
                            desired_joints = map_joint_target(
                                joint_anchor, selected_joint, dx,
                                args.joint_scale_deg, args.joint_max_from_anchor_deg,
                            )
                            if now - last_update >= 1.0 / args.update_hz:
                                lara.set_joint_target(desired_joints)
                                last_update = now
                                if now - last_motion_log >= 1.0:
                                    print(
                                        f"LARA servo_j: joint=J{selected_joint} "
                                        f"rate={lara.joint_stream_rate:.0f}Hz "
                                        f"response={lara.last_servo_response}"
                                    )
                                    last_motion_log = now
                            motion_status = f"PINCH: J{selected_joint}"
                        pinch_previous = True
                    else:
                        reset_motion_anchors()
                        if rearm_after_mode_change:
                            motion_status = "RELEASE PINCH"
                        elif control_mode == "CARTESIAN" and axis_mode == "PAUSE":
                            motion_status = "LEFT HAND: PAUSE"
                        elif control_mode == "JOINT" and selected_joint is None:
                            motion_status = "SELECT J1-J6"
                        else:
                            motion_status = "READY - PINCH TO MOVE"
                else:
                    filtered_palm = None
                    filtered_hand_size = None
                    reset_motion_anchors()

                if lara.armed and now - last_update >= 1.0 / args.update_hz:
                    lara.hold_mode(control_mode)
                    last_update = now

                if manual_stop:
                    mode_text = "SOFTWARE STOP - A ZUM FREIGEBEN"
                else:
                    wait_text = (
                        "WARTE AUF BEIDE HAENDE"
                        if args.require_axis_hand
                        else "WARTE AUF RECHTE STEUERHAND"
                    )
                    mode_text = "AKTIV" if lara.armed else wait_text
                joint_text = f"J{selected_joint}" if selected_joint else "-"
                core.put_status(
                    frame,
                    (
                        (f"{mode_text} | {'HARDWARE' if args.execute else 'CAMERA TEST'}", (0, 220, 0) if lara.armed else (0, 0, 255)),
                        (f"{control_mode} | ACHSE {axis_mode} | JOINT {joint_text}", (0, 200, 255)),
                        (f"{motion_status} | PINCH {pinch_ratio:.2f}", (0, 220, 0) if pinch_active else (220, 220, 220)),
                        (f"GRIPPER {gripper_state} | WAYPOINT {waypoint_status}", (255, 200, 0)),
                        ("WORKSPACE LIMIT" if workspace_limited else "Q: quit", (0, 0, 255) if workspace_limited else (220, 220, 220)),
                    ),
                )
                cv2.imshow("LARA + Contactile Mimic - 2 Hands", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord("s"), ord("S"), ord(" ")):
                    manual_stop = True
                    reset_motion_anchors()
                    if lara.armed:
                        lara.set_armed(False)
                    print("SOFTWARE STOP: Servo-Schnittstelle deaktiviert und LARA gestoppt.")
                if key in (ord("a"), ord("A")):
                    if not manual_stop:
                        print("LARA ist bereits freigegeben.")
                    elif not required_hands_present:
                        required_text = (
                            "Beide Haende muessen sichtbar sein."
                            if args.require_axis_hand
                            else "Die rechte Steuerhand muss sichtbar sein."
                        )
                        print(f"Freigabe verweigert: {required_text}")
                    elif pinch_active:
                        print("Freigabe verweigert: Rechten Pinch zuerst loesen.")
                    else:
                        manual_stop = False
                        lara.set_armed(True)
                        last_required_hands_seen = now
                        reset_motion_anchors()
                        print("LARA wieder freigegeben. Pinch zum Bewegen schliessen.")
                if key == ord("t"):
                    save_waypoint(
                        waypoints, lara, gripper_state, args.waypoint_file,
                        args.waypoint_min_distance,
                    )
    finally:
        lara.close()
        camera.release()
        cv2.destroyAllWindows()


def main(argv: Sequence[str] | None = None) -> int:
    core.install_cleanup_signal_handlers()
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
