#!/usr/bin/env python3
"""
uFactory 850 & xArm Gripper Direct Real Robot Controller - Standalone (No RoboDK)

Purpose:
    Direct vision teleoperation for uFactory 850 and xArm Gripper using MediaPipe and xArm-Python-SDK.
    Operates completely without RoboDK, communicating directly over Ethernet TCP/IP to the robot controller.

Architecture (Mimic Style):
    - Dry-Run / Camera Preview Mode by default (safe testing without robot).
    - Physical robot motion enabled ONLY with the --execute flag.
    - Right Hand (Dead-Man Motion Clutch & Gripper):
        * Thumb + Index Pinch: Commands movement relative to pinch start anchor.
        * Release Pinch: Immediately freezes/holds position.
        * Closed Fist (Hold for 0.7s): Toggles xArm Gripper between OPEN and CLOSE.
        * V-Sign (Peace Sign): Saves current pose to a JSON waypoint file.
    - Left Hand (Axis Locks & Modes):
        * Fist: PAUSE (motion locked)
        * 1 Finger: X-Axis only
        * 2 Fingers: Y-Axis only
        * 3 Fingers: Z-Axis only
        * 4 Fingers (Open Palm): FREE 3D motion
        * Thumbs-Up / Down: Switch between CARTESIAN and JOINT modes
        * Finger Count (1 to 6): Select joint J1 to J6 in JOINT mode
    - Safety Controls:
        * Dynamic workspace bounding box around activation pose
        * Velocity and step rate clamping
        * Zero-jump anchor reset upon each new pinch engagement
        * Safe clean shutdown on Q / ESC / SIGINT
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
except ImportError:
    mp = None

# Optional direct real-time driver for uFactory xArm
try:
    from xarm.wrapper import XArmAPI
except ImportError:
    XArmAPI = None


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

CONTROL_MODES = ("CARTESIAN", "JOINT")
AXIS_MODES = ("PAUSE", "X", "Y", "Z", "FREE")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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


def finger_extended(
    landmarks: Sequence[Any],
    tip_index: int,
    pip_index: int,
    center: tuple[float, float],
) -> bool:
    tip_dist = math.hypot(landmarks[tip_index].x - center[0], landmarks[tip_index].y - center[1])
    pip_dist = math.hypot(landmarks[pip_index].x - center[0], landmarks[pip_index].y - center[1])
    return tip_dist > pip_dist * 1.25


def count_extended_fingers(landmarks: Sequence[Any]) -> int:
    center = palm_center(landmarks)
    return sum(
        finger_extended(landmarks, tip, pip, center)
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
    )


def detect_axis_mode(landmarks: Sequence[Any]) -> str:
    count = count_extended_fingers(landmarks)
    mapping = {0: "PAUSE", 1: "X", 2: "Y", 3: "Z", 4: "FREE"}
    return mapping.get(count, "FREE")


def detect_mode_switch_gesture(landmarks: Sequence[Any], margin: float = 0.04) -> str | None:
    """Thumb Up = CARTESIAN, Thumb Down = JOINT."""
    if count_extended_fingers(landmarks) != 0:
        return None
    center = palm_center(landmarks)
    tip, mcp = landmarks[4], landmarks[2]
    tip_dist = math.hypot(tip.x - center[0], tip.y - center[1])
    mcp_dist = math.hypot(mcp.x - center[0], mcp.y - center[1])
    if tip_dist <= mcp_dist * 1.2:
        return None
    if tip.y < mcp.y - margin:
        return "CARTESIAN"
    if tip.y > mcp.y + margin:
        return "JOINT"
    return None


def detect_joint_selection(landmarks: Sequence[Any]) -> int | None:
    center = palm_center(landmarks)
    idx = finger_extended(landmarks, 8, 6, center)
    mid = finger_extended(landmarks, 12, 10, center)
    rng = finger_extended(landmarks, 16, 14, center)
    pnk = finger_extended(landmarks, 20, 18, center)

    if not any((idx, mid, rng, pnk)):
        return 6
    if idx and not any((mid, rng, pnk)):
        return 1
    if idx and mid and not any((rng, pnk)):
        return 2
    if idx and mid and rng and not pnk:
        return 3
    if idx and mid and rng and pnk:
        thumb_dist = distance_2d(landmarks[4], landmarks[5])
        palm_w = max(0.001, distance_2d(landmarks[5], landmarks[17]))
        return 5 if (thumb_dist / palm_w) > 0.75 else 4
    return None


def is_v_sign(landmarks: Sequence[Any]) -> bool:
    center = palm_center(landmarks)
    idx = finger_extended(landmarks, 8, 6, center)
    mid = finger_extended(landmarks, 12, 10, center)
    rng = finger_extended(landmarks, 16, 14, center)
    pnk = finger_extended(landmarks, 20, 18, center)
    palm_w = max(0.001, distance_2d(landmarks[5], landmarks[17]))
    separated = distance_2d(landmarks[8], landmarks[12]) > palm_w * 0.35
    return idx and mid and not rng and not pnk and separated


def get_hands_by_label(result: Any, swap_hand_labels: bool = True) -> dict[str, Any]:
    hands: dict[str, Any] = {}
    for landmarks, handedness in zip(result.hand_landmarks or (), result.handedness or ()):
        if not handedness:
            continue
        label = handedness[0].category_name
        if swap_hand_labels:
            label = {"Left": "Right", "Right": "Left"}.get(label, label)
        hands[label] = landmarks
    return hands


# ============================================================
# DIRECT uFACTORY 850 CONTROLLER CLASS (via XArmAPI)
# ============================================================

@dataclass
class MotionResult:
    pose: list[float]
    limited: bool


class DirectXArmController:
    """Direct standalone interface to uFactory 850 using official xarm-python-sdk."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.arm = None
        self.armed = False
        # xArm TCP pose format: [X, Y, Z, Roll, Pitch, Yaw] in millimeters and degrees
        self.commanded_pose = [300.0, 0.0, 250.0, 180.0, 0.0, 0.0]
        self.commanded_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.workspace_center = list(self.commanded_pose[:3])
        self.joint_workspace_center = list(self.commanded_joints)
        self.gripper_state = "OPEN"
        self.last_command_time = time.monotonic()

    def connect(self) -> None:
        if not self.args.execute:
            print("[DRY-RUN] Simulation mode: Physical uFactory 850 will NOT be commanded.")
            print("[DRY-RUN] Use --execute --robot-ip <IP> for real robot control.")
            return

        if XArmAPI is None:
            raise RuntimeError(
                "Package 'xarm-python-sdk' is not installed in the active Python environment.\n"
                "To connect to the physical uFactory 850 without RoboDK, install it with:\n"
                "    pip install xarm-python-sdk"
            )

        print(f"Connecting directly to uFactory 850 at {self.args.robot_ip}...")
        self.arm = XArmAPI(self.args.robot_ip, is_radian=False)
        self.arm.clean_warn()
        self.arm.clean_error()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(1)   # Mode 1 = Servo Cartesian/Joint streaming
        self.arm.set_state(state=0)  # State 0 = Sport/Ready state

        # Read actual robot state
        code, pos = self.arm.get_position(is_radian=False)
        if code == 0 and pos and len(pos) >= 6:
            self.commanded_pose = [float(v) for v in pos[:6]]
            self.workspace_center = list(self.commanded_pose[:3])

        code, joints = self.arm.get_servo_angle(is_radian=False)
        if code == 0 and joints and len(joints) >= 6:
            self.commanded_joints = [float(v) for v in joints[:6]]
            self.joint_workspace_center = list(self.commanded_joints)

        # Initialize gripper if enabled
        if self.args.gripper:
            self.arm.set_gripper_enable(True)
            self.arm.set_gripper_mode(0)
            self.set_gripper("OPEN")

        print(f"Connected to uFactory 850 successfully!")
        print(f"Current TCP (mm, deg): {[round(v, 2) for v in self.commanded_pose]}")
        print(f"Current Joints (deg): {[round(v, 2) for v in self.commanded_joints]}")

    def read_current_state(self) -> None:
        if self.arm is not None and self.arm.connected:
            code, pos = self.arm.get_position(is_radian=False)
            if code == 0 and pos:
                self.commanded_pose = [float(v) for v in pos[:6]]
            code, joints = self.arm.get_servo_angle(is_radian=False)
            if code == 0 and joints:
                self.commanded_joints = [float(v) for v in joints[:6]]

    def reanchor(self) -> None:
        self.read_current_state()
        self.workspace_center = list(self.commanded_pose[:3])
        self.joint_workspace_center = list(self.commanded_joints)

    def send_cartesian_pose(self, desired_xyz_mm: Sequence[float]) -> MotionResult:
        """Sends safe, clamped set_servo_cartesian command in mm."""
        bounds = (self.args.workspace_x * 1000.0, self.args.workspace_y * 1000.0, self.args.workspace_z * 1000.0)
        target = list(self.commanded_pose)
        limited = False

        for i in range(3):
            low = self.workspace_center[i] - bounds[i]
            high = self.workspace_center[i] + bounds[i]
            val = clamp(desired_xyz_mm[i], low, high)
            if not math.isclose(val, desired_xyz_mm[i], abs_tol=1e-3):
                limited = True
            target[i] = val

        dt = max(0.001, time.monotonic() - self.last_command_time)
        max_step = min(self.args.max_tcp_step * 1000.0, self.args.max_speed * 1000.0 * dt)

        delta = [(target[i] - self.commanded_pose[i]) * self.args.smoothing for i in range(3)]
        dist = math.hypot(delta[0], delta[1], delta[2])
        if dist > max_step and dist > 1e-6:
            scale = max_step / dist
            delta = [d * scale for d in delta]

        target[:3] = [self.commanded_pose[i] + delta[i] for i in range(3)]

        if self.arm is not None and self.args.execute:
            # set_servo_cartesian(mvpose, is_radian=False)
            self.arm.set_servo_cartesian(target, is_radian=False)

        self.commanded_pose = target
        self.last_command_time = time.monotonic()
        return MotionResult(target, limited)

    def send_joint_delta(self, joint_idx: int, delta_deg: float) -> None:
        """Move a single joint in JOINT mode with set_servo_angle_j."""
        if joint_idx < 1 or joint_idx > 6:
            return
        idx = joint_idx - 1
        limit_deg = self.args.joint_workspace_deg
        center = self.joint_workspace_center[idx]
        target_q = list(self.commanded_joints)
        new_val = clamp(target_q[idx] + delta_deg, center - limit_deg, center + limit_deg)
        target_q[idx] = new_val

        if self.arm is not None and self.args.execute:
            self.arm.set_servo_angle_j(target_q, is_radian=False)

        self.commanded_joints = target_q
        self.last_command_time = time.monotonic()

    def set_gripper(self, state: str) -> None:
        self.gripper_state = state
        if self.arm is not None and self.args.execute and self.args.gripper:
            # 840 = fully open, 0 = fully closed
            pos = 840.0 if state == "OPEN" else 0.0
            self.arm.set_gripper_position(pos, wait=False)
        print(f"[GRIPPER] State set to: {state}")

    def hold(self) -> None:
        pass  # In servo mode, omitting new packets naturally holds current setpoint

    def close(self) -> None:
        print("\nStopping uFactory 850 direct controller...")
        if self.arm is not None:
            try:
                self.arm.set_state(4)  # Stop state
                self.arm.motion_enable(enable=False)
                self.arm.disconnect()
            except Exception as e:
                print(f"Notice on xArm disconnect: {e}")
        print("uFactory 850 interface closed.")


# ============================================================
# MAIN APPLICATION LOOP
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct uFactory 850 Vision Controller (No RoboDK)")
    parser.add_argument("--execute", action="store_true", help="Connect to physical uFactory 850 (default: Dry-run preview)")
    parser.add_argument("--robot-ip", type=str, default="192.168.1.200", help="uFactory controller IP address")
    parser.add_argument("--gripper", action="store_true", default=True, help="Enable xArm Gripper control")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index")
    parser.add_argument("--model-path", type=str, default="hand_landmarker.task", help="Path to hand_landmarker.task")

    # Workspace Limits (meters)
    parser.add_argument("--workspace-x", type=float, default=0.25, help="Cartesian travel limit +-X (m)")
    parser.add_argument("--workspace-y", type=float, default=0.25, help="Cartesian travel limit +-Y (m)")
    parser.add_argument("--workspace-z", type=float, default=0.20, help="Cartesian travel limit +-Z (m)")
    parser.add_argument("--joint-workspace-deg", type=float, default=25.0, help="Joint limit +-deg in JOINT mode")

    # Speed & Motion Scaling
    parser.add_argument("--max-speed", type=float, default=0.08, help="Max TCP linear speed (m/s)")
    parser.add_argument("--max-tcp-step", type=float, default=0.010, help="Max TCP displacement per frame (m)")

    # Sensitivity
    parser.add_argument("--horizontal-scale", type=float, default=0.50, help="Meters per full camera width")
    parser.add_argument("--vertical-scale", type=float, default=0.40, help="Meters per full camera height")
    parser.add_argument("--depth-scale", type=float, default=0.35, help="Meters per palm size change")
    parser.add_argument("--smoothing", type=float, default=0.30, help="Low-pass filter alpha (0.1 to 0.8)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    controller = DirectXArmController(args)
    try:
        controller.connect()
    except Exception as e:
        print(f"\n[ERROR] Connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize MediaPipe
    if mp is None or cv2 is None:
        print("[ERROR] OpenCV and MediaPipe are required. Run: pip install -r requirements.txt")
        sys.exit(1)

    model_file = Path(args.model_path)
    if not model_file.exists():
        model_file = Path(__file__).parent / "hand_landmarker.task"
    if not model_file.exists():
        print(f"[ERROR] Landmark model not found at {model_file}")
        sys.exit(1)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_file)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {args.camera_index}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(" uFACTORY 850 DIRECT VISION CONTROLLER (NO ROBODK)")
    print(f" Mode: {'LIVE PHYSICAL uFACTORY 850' if args.execute else 'DRY-RUN PREVIEW'}")
    print(" Right Hand: Pinch to clutch move | Closed Fist (0.7s) to toggle Gripper")
    print(" Left Hand:  Fist=PAUSE | 1=X | 2=Y | 3=Z | 4=FREE | Thumbs=Mode")
    print(" Keys: 'R' Re-anchor pose | 'Q' / ESC Quit")
    print("=" * 60 + "\n")

    # State
    pinch_engaged = False
    control_mode = "CARTESIAN"
    axis_mode = "PAUSE"
    selected_joint = 1
    anchor_tcp = list(controller.commanded_pose[:3])
    anchor_palm_x, anchor_palm_y, anchor_palm_w = 0.5, 0.5, 0.15
    v_sign_start_time = None
    fist_hold_start = None
    saved_waypoints: list[dict[str, Any]] = []

    landmarker = HandLandmarker.create_from_options(options)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab camera frame.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            timestamp_ms = int(time.time() * 1000)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            hands = get_hands_by_label(result, swap_hand_labels=True)
            right_hand = hands.get("Right")
            left_hand = hands.get("Left")

            # ----------------------------------------------------
            # 1. Process Left Hand (Mode & Axis Locks)
            # ----------------------------------------------------
            if left_hand:
                mode_gesture = detect_mode_switch_gesture(left_hand)
                if mode_gesture:
                    control_mode = mode_gesture

                if control_mode == "CARTESIAN":
                    axis_mode = detect_axis_mode(left_hand)
                else:
                    j = detect_joint_selection(left_hand)
                    if j is not None:
                        selected_joint = j

            # ----------------------------------------------------
            # 2. Process Right Hand (Motion, Gripper & Waypoints)
            # ----------------------------------------------------
            if right_hand:
                palm_cx, palm_cy = palm_center(right_hand)
                palm_w = max(0.001, distance_2d(right_hand[5], right_hand[17]))
                pinch_dist = distance_2d(right_hand[4], right_hand[8]) / palm_w
                is_pinching = pinch_dist < 0.40
                fingers_open = count_extended_fingers(right_hand)

                # Gripper toggle via closed fist held for 0.7s (when not pinching)
                if fingers_open == 0 and not is_pinching:
                    if fist_hold_start is None:
                        fist_hold_start = time.monotonic()
                    elif time.monotonic() - fist_hold_start > 0.70:
                        new_state = "CLOSE" if controller.gripper_state == "OPEN" else "OPEN"
                        controller.set_gripper(new_state)
                        fist_hold_start = time.monotonic() + 1.0  # Cooldown
                else:
                    if fist_hold_start and time.monotonic() > fist_hold_start:
                        fist_hold_start = None

                # Waypoint recording via V-Sign
                if is_v_sign(right_hand):
                    if v_sign_start_time is None:
                        v_sign_start_time = time.monotonic()
                    elif time.monotonic() - v_sign_start_time > 0.50:
                        waypoint = {
                            "time": datetime.now().isoformat(),
                            "tcp_pose_mm": list(controller.commanded_pose),
                            "joints_deg": list(controller.commanded_joints),
                            "gripper": controller.gripper_state,
                        }
                        saved_waypoints.append(waypoint)
                        with open("xarm_waypoints.json", "w") as f:
                            json.dump(saved_waypoints, f, indent=2)
                        print(f"Waypoint #{len(saved_waypoints)} saved to xarm_waypoints.json!")
                        v_sign_start_time = time.monotonic() + 1.0
                else:
                    v_sign_start_time = None

                # Pinch state machine (Zero-Jump Anchoring)
                if is_pinching and not pinch_engaged:
                    pinch_engaged = True
                    controller.reanchor()
                    anchor_tcp = list(controller.commanded_pose[:3])
                    anchor_palm_x, anchor_palm_y, anchor_palm_w = palm_cx, palm_cy, palm_w
                    print("Pinch ENGAGED -> Zero-jump anchor set.")

                elif not is_pinching and pinch_engaged:
                    pinch_engaged = False
                    controller.hold()
                    print("Pinch RELEASED -> Robot position held.")

                # Move if engaged and axis is not PAUSE
                if pinch_engaged and axis_mode != "PAUSE":
                    dx_mm = (palm_cx - anchor_palm_x) * args.horizontal_scale * 1000.0
                    dy_mm = ((palm_w - anchor_palm_w) / anchor_palm_w) * args.depth_scale * 1000.0
                    dz_mm = -(palm_cy - anchor_palm_y) * args.vertical_scale * 1000.0

                    if control_mode == "CARTESIAN":
                        desired = list(anchor_tcp)
                        if axis_mode in ("X", "FREE"):
                            desired[0] += dx_mm
                        if axis_mode in ("Y", "FREE"):
                            desired[1] += dy_mm
                        if axis_mode in ("Z", "FREE"):
                            desired[2] += dz_mm

                        controller.send_cartesian_pose(desired)

                    elif control_mode == "JOINT":
                        joint_delta_deg = (palm_cx - anchor_palm_x) * 15.0
                        controller.send_joint_delta(selected_joint, joint_delta_deg)

            else:
                if pinch_engaged:
                    pinch_engaged = False
                    controller.hold()

            # ----------------------------------------------------
            # 3. Visual On-Screen HUD Overlay
            # ----------------------------------------------------
            color = (0, 255, 0) if not args.execute else (0, 0, 255)
            status = "DRY-RUN (Sim)" if not args.execute else "LIVE uFACTORY 850"

            cv2.rectangle(frame, (10, 10), (500, 160), (20, 20, 20), -1)
            cv2.putText(frame, f"uFactory Direct: {status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Mode: {control_mode} | Axis: {axis_mode}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Clutch: {'ENGAGED' if pinch_engaged else 'DISENGAGED'} | Gripper: {controller.gripper_state}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if pinch_engaged else (180, 180, 180), 1)
            cv2.putText(frame, f"Joint: J{selected_joint} | Saved Waypoints: {len(saved_waypoints)}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("uFactory 850 Direct Vision Teleop (No RoboDK)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('r'), ord('R')):
                controller.reanchor()
                print("Pose re-anchored manually.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        controller.close()


if __name__ == "__main__":
    main()
