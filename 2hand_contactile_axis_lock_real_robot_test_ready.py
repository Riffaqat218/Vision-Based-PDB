"""
LARA8 2-Hand Contactile Axis-Lock Webcam Controller - Real Robot Test Ready Version

Purpose:
    - 2-Hand tracking with MediaPipe HandLandmarker.
    - Right hand (Control Hand): Pinch gesture controls LARA8 movement through RoboDK IK.
    - Left hand (Axis Hand): Selects Cartesian axis locks (PAUSE, X, Y, Z, FREE) or selects Joints (J1-J6).
    - Left hand Thumbs Up / Down: Switch between CARTESIAN and JOINT control modes.
    - Right hand V-sign gesture: Gesture-based waypoint saving.
    - Right hand Open Palm / Closed Fist: Controls the Contactile Gripper (GAL2_MASTER).
    - Pick/drop code is disabled for initial real robot testing.

IMPORTANT REAL ROBOT NOTES:
    1. Test in RoboDK simulation first (keep REAL_ROBOT_MODE = False).
    2. Keep the physical robot E-stop reachable at all times.
    3. Use slow robot speed and small workspace limits for first real test.
    4. Camera must be fixed; do not move the camera after tuning axes.
    5. Set REAL_ROBOT_MODE = True only when the RoboDK station is connected to the real robot.
    6. Real gripper commands require setting REAL_GRIPPER_COMMANDS_ENABLED = True and
       implementing send_real_gripper_command() for your physical hardware API / Modbus / I/O.
"""

import math
import time
from pathlib import Path
from copy import deepcopy
import csv
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

from robodk.robolink import (
    Robolink,
    ITEM_TYPE_FRAME,
    ITEM_TYPE_TARGET,
    ITEM_TYPE_ROBOT,
    ITEM_TYPE_TOOL,
)
from robodk.robomath import eye, transl

# ============================================================
# CONFIGURATION
# ============================================================

# ============================================================
# REAL ROBOT TEST SWITCHES
# ============================================================

# Keep this False for simulation. Set True only when the RoboDK station is
# connected to the real LARA8 robot through the correct RoboDK robot driver.
REAL_ROBOT_MODE = False

# Real gripper commands are not enabled by default. Set True once physical
# gripper communication (Modbus, URCap, controller I/O, or vendor API) is configured.
REAL_GRIPPER_COMMANDS_ENABLED = False

# Model path for MediaPipe HandLandmarker
MODEL_PATH = Path("hand_landmarker.task")

CAMERA_INDEX = 0

# Station tree names (must match RoboDK station tree)
TARGET_NAME = "Hand_Target"
BASE_FRAME_NAME = "Neura LARA 8 Base"
ROBOT_NAME = "Neura LARA 8"
GRIPPER_NAME = "GAL2_MASTER"
PICK_OBJECT_NAME = "Sphere D100mm"

# Workspace & displacement limits
MAX_X_MOVEMENT_MM = 250.0
MAX_Z_MOVEMENT_MM = 250.0

X_SCALE_MM = 750.0
Z_SCALE_MM = 400.0
Z_UP_SCALE_MM = 450.0
Z_DOWN_SCALE_MM = 1200.0

SMOOTHING_ALPHA = 0.30
PINCH_THRESHOLD = 0.75
DEAD_ZONE = 0.004

# Robot joint & motion safety settings
MAX_JOINT_STEP_DEG = 20.0
ROBOT_UPDATE_INTERVAL = 0.035
TARGET_SMOOTHING_ALPHA = 0.28
MAX_TCP_STEP_MM = 8.0
MIN_TCP_STEP_MM = 0.2

# Safe workspace box around initial TCP position (mm)
WORKSPACE_X_NEG_MM = 350.0
WORKSPACE_X_POS_MM = 350.0
WORKSPACE_Y_NEG_MM = 250.0
WORKSPACE_Y_POS_MM = 250.0
WORKSPACE_Z_NEG_MM = 550.0
WORKSPACE_Z_POS_MM = 300.0
CLAMP_TO_WORKSPACE = True

# Webcam depth estimation
ENABLE_WEBCAM_DEPTH = True
Y_SCALE_MM = 220.0
MAX_Y_MOVEMENT_MM = 100.0
DEPTH_SMOOTHING_ALPHA = 0.15
Y_DIRECTION = 1.0

# Path Recording
ENABLE_PATH_RECORDING = False
RECORD_INTERVAL_SEC = 0.25
RECORD_MIN_TCP_DISTANCE_MM = 15.0
MAX_RECORDED_POINTS = 40
RECORDINGS_FOLDER = Path("recordings")
RECORDED_PROGRAM_NAME = "Hand_Guided_LARA8_Path"

# Waypoint Teaching
ENABLE_MANUAL_WAYPOINT_TEACHING = True
TEACH_TARGET_PREFIX = "Hand_Taught_Target"
TEACH_PROGRAM_NAME = "Hand_Taught_Program"

# Contactile Gripper Settings
ENABLE_GRIPPER_GESTURES = True
GRIPPER_OPEN_VALUE = 35.0
GRIPPER_CLOSED_VALUE = 0.0
GRIPPER_DEBOUNCE_FRAMES = 8
GRIPPER_STEP = 0.08

# Disabled for initial real robot testing safety
ENABLE_PICK_AND_DROP = False
GRIP_ATTACH_FRAME_NAME = "Grip_Attach_Frame"
ATTACH_DISTANCE_MM = 150.0
CUBE_HOLD_OFFSET_MM = [0, 0, 0]

# Gesture-based Waypoint Saving
ENABLE_GESTURE_WAYPOINT_SAVE = True
SAVE_GESTURE_HOLD_SEC = 0.50
SAVE_TARGET_MIN_DISTANCE_MM = 20.0
V_SIGN_SEPARATION_RATIO = 0.20

# Contact items for adaptive collision detection (simulation)
CONTACT_LEFT_ITEM_NAME = "link1_mech"
CONTACT_RIGHT_ITEM_NAME = "link2_mech"
CONTACT_CLOSE_STEP = 0.01
CONTACT_SETTLE_SEC = 0.02

# 2-Hand Control Configuration
CONTROL_HAND_LABEL = "Right"
AXIS_HAND_LABEL = "Left"
SWAP_HAND_LABELS = True
REQUIRE_AXIS_HAND = True
AXIS_MODE_DEFAULT = "PAUSE"
AXIS_DEBOUNCE_FRAMES = 6

# Mode selection
CONTROL_MODE_DEFAULT = "CARTESIAN"
MODE_SWITCH_HOLD_SEC = 0.50
THUMB_MODE_MARGIN = 0.04

# Joint mode settings
JOINT_SELECT_DEBOUNCE_FRAMES = 6
JOINT_HAND_SCALE_DEG = 180.0
JOINT_MAX_FROM_ANCHOR_DEG = 45.0
JOINT_CONTROL_STEP_DEG = 2.0
JOINT_UPDATE_INTERVAL = 0.035

# Hand Skeleton Connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_contactile_gripper(rdk):
    """Load Contactile gripper master item from RoboDK."""
    gripper = rdk.Item(GRIPPER_NAME)
    if not gripper.Valid():
        print(f'Warning: Contactile master "{GRIPPER_NAME}" was not found.')
        return None
    print(f'Loaded Contactile gripper master: "{gripper.Name()}"')
    return gripper


def distance_2d(point_a, point_b):
    """Return the 2D distance between two normalized landmarks."""
    return math.hypot(
        point_a.x - point_b.x,
        point_a.y - point_b.y,
    )


def calculate_palm_center(landmarks):
    """Stable palm center using wrist and knuckle points."""
    weighted_points = [
        (0, 0.18),    # wrist
        (1, 0.08),    # thumb base
        (5, 0.17),    # index base
        (9, 0.22),    # middle base
        (13, 0.18),   # ring base
        (17, 0.17),   # little base
    ]
    x = sum(landmarks[i].x * w for i, w in weighted_points)
    y = sum(landmarks[i].y * w for i, w in weighted_points)
    return np.array([x, y], dtype=np.float64)


def draw_hand(frame, landmarks, label="", line_color=(255, 255, 255)):
    """Draw one detected hand skeleton and role label."""
    height, width = frame.shape[:2]
    points = []
    for landmark in landmarks:
        pixel_x = int(landmark.x * width)
        pixel_y = int(landmark.y * height)
        points.append((pixel_x, pixel_y))

    for start_index, end_index in HAND_CONNECTIONS:
        cv2.line(frame, points[start_index], points[end_index], line_color, 2)

    for index, point in enumerate(points):
        color = (0, 255, 255)
        if index == 4:
            color = (255, 0, 255)   # thumb tip
        elif index == 8:
            color = (0, 255, 0)     # index tip
        cv2.circle(frame, point, 5, color, -1)

    if label and points:
        label_position = (
            max(10, points[0][0] - 35),
            max(25, points[0][1] + 30),
        )
        cv2.putText(
            frame, label, label_position,
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, line_color, 2
        )


def get_hands_by_label(result):
    """Return detected hands using MediaPipe handedness labels."""
    hands = {}
    if not result.hand_landmarks:
        return hands

    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        if not handedness:
            continue
        label = handedness[0].category_name
        if SWAP_HAND_LABELS:
            if label == "Left":
                label = "Right"
            elif label == "Right":
                label = "Left"
        hands[label] = landmarks
    return hands


def is_finger_extended(landmarks, tip_index, pip_index, palm_center):
    """Checks if a finger is extended relative to palm center."""
    tip_distance = math.hypot(
        landmarks[tip_index].x - palm_center[0],
        landmarks[tip_index].y - palm_center[1],
    )
    pip_distance = math.hypot(
        landmarks[pip_index].x - palm_center[0],
        landmarks[pip_index].y - palm_center[1],
    )
    return tip_distance > pip_distance * 1.25


def count_extended_fingers(landmarks):
    """Count extended index, middle, ring, and little fingers."""
    palm_center = calculate_palm_center(landmarks)
    return sum([
        is_finger_extended(landmarks, 8, 6, palm_center),
        is_finger_extended(landmarks, 12, 10, palm_center),
        is_finger_extended(landmarks, 16, 14, palm_center),
        is_finger_extended(landmarks, 20, 18, palm_center),
    ])


def detect_control_mode_gesture(landmarks):
    """
    Left-hand mode-selection gestures:
        thumbs up   -> CARTESIAN
        thumbs down -> JOINT
    """
    if count_extended_fingers(landmarks) != 0:
        return None

    palm_center = calculate_palm_center(landmarks)
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]

    thumb_tip_distance = math.hypot(
        thumb_tip.x - palm_center[0],
        thumb_tip.y - palm_center[1],
    )
    thumb_mcp_distance = math.hypot(
        thumb_mcp.x - palm_center[0],
        thumb_mcp.y - palm_center[1],
    )

    if thumb_tip_distance <= thumb_mcp_distance * 1.25:
        return None

    if thumb_tip.y < thumb_mcp.y - THUMB_MODE_MARGIN:
        return "CARTESIAN"
    if thumb_tip.y > thumb_mcp.y + THUMB_MODE_MARGIN:
        return "JOINT"

    return None


def detect_axis_mode(landmarks):
    """
    Left-hand axis-selection gestures:
        fist / 0 fingers  -> PAUSE
        1 finger          -> X ONLY
        2 fingers         -> Y ONLY
        3 fingers         -> Z ONLY
        4 fingers         -> FREE XYZ
    """
    extended_count = count_extended_fingers(landmarks)
    axis_modes = {
        0: "PAUSE",
        1: "X",
        2: "Y",
        3: "Z",
        4: "FREE",
    }
    return axis_modes.get(extended_count)


def is_thumb_extended_for_joint(landmarks):
    """Detect whether the thumb is clearly spread away from palm."""
    palm_width = distance_2d(landmarks[5], landmarks[17])
    if palm_width < 0.001:
        return False
    thumb_distance = distance_2d(landmarks[4], landmarks[5])
    return (thumb_distance / palm_width) > 0.75


def detect_joint_selection_gesture(landmarks):
    """
    Left-hand joint-selection gestures:
        1 finger  -> J1
        2 fingers -> J2
        3 fingers -> J3
        4 fingers (thumb folded) -> J4
        5 fingers (open palm)    -> J5
        Closed fist              -> J6
    """
    palm_center = calculate_palm_center(landmarks)
    index_ext = is_finger_extended(landmarks, 8, 6, palm_center)
    middle_ext = is_finger_extended(landmarks, 12, 10, palm_center)
    ring_ext = is_finger_extended(landmarks, 16, 14, palm_center)
    little_ext = is_finger_extended(landmarks, 20, 18, palm_center)
    thumb_ext = is_thumb_extended_for_joint(landmarks)

    if not index_ext and not middle_ext and not ring_ext and not little_ext:
        return 6
    if index_ext and not middle_ext and not ring_ext and not little_ext:
        return 1
    if index_ext and middle_ext and not ring_ext and not little_ext:
        return 2
    if index_ext and middle_ext and ring_ext and not little_ext:
        return 3
    if index_ext and middle_ext and ring_ext and little_ext:
        return 5 if thumb_ext else 4

    return None


def apply_axis_lock(robot_dx, robot_dy, robot_dz, axis_mode):
    """Filter displacement based on active axis mode."""
    if axis_mode == "X":
        return robot_dx, 0.0, 0.0
    if axis_mode == "Y":
        return 0.0, robot_dy, 0.0
    if axis_mode == "Z":
        return 0.0, 0.0, robot_dz
    if axis_mode == "FREE":
        return robot_dx, robot_dy, robot_dz
    return 0.0, 0.0, 0.0


def connect_to_robodk():
    """Connect to RoboDK and retrieve items."""
    rdk = Robolink()
    robot = rdk.Item(ROBOT_NAME, ITEM_TYPE_ROBOT)
    target = rdk.Item(TARGET_NAME, ITEM_TYPE_TARGET)
    base_frame = rdk.Item(BASE_FRAME_NAME, ITEM_TYPE_FRAME)

    if not robot.Valid():
        raise RuntimeError(f'RoboDK robot "{ROBOT_NAME}" was not found.')
    if not target.Valid():
        raise RuntimeError(f'RoboDK target "{TARGET_NAME}" was not found.')
    if not base_frame.Valid():
        raise RuntimeError(f'RoboDK frame "{BASE_FRAME_NAME}" was not found.')

    if not target.Parent().equals(base_frame):
        target.setParentStatic(base_frame)

    target.setAsCartesianTarget()
    robot.setPoseFrame(base_frame)
    return rdk, robot, target, base_frame


def wrapped_joint_difference(new_value, current_value):
    """Shortest angular difference wrapping around +/- 180 deg."""
    return (new_value - current_value + 180.0) % 360.0 - 180.0


def get_joint_solutions_from_mat(solutions_mat):
    """Convert RoboDK matrix to list of joint lists."""
    solutions = []
    rows = solutions_mat.size(0)
    cols = solutions_mat.size(1)
    for col in range(cols):
        joint_solution = [float(solutions_mat[row, col]) for row in range(rows)]
        solutions.append(joint_solution)
    return solutions


def joint_distance(solution_joints, current_joints):
    """Calculates angular distance between solutions and current joints."""
    total = 0.0
    for sol_j, cur_j in zip(solution_joints, current_joints):
        diff = wrapped_joint_difference(sol_j, cur_j)
        total += diff * diff
    return math.sqrt(total)


def limit_joint_step(current_joints, target_joints, max_step_deg):
    """Limit joint step per update cycle."""
    new_joints = []
    for cur_j, tgt_j in zip(current_joints, target_joints):
        diff = wrapped_joint_difference(tgt_j, cur_j)
        diff_limited = float(np.clip(diff, -max_step_deg, max_step_deg))
        new_joints.append(cur_j + diff_limited)
    return new_joints


def send_real_gripper_command(opening):
    """
    REAL ROBOT GRIPPER PLACEHOLDER.
    opening = 1.0 means open
    opening = 0.0 means closed

    Replace this with your physical gripper communication (Modbus TCP,
    robot controller digital output, vendor API, etc.).
    """
    if not REAL_ROBOT_MODE or not REAL_GRIPPER_COMMANDS_ENABLED:
        return

    print(f"REAL GRIPPER COMMAND SENT: opening={opening:.2f}")


def move_robot_to_target(robot, target_pose, tool_pose):
    """
    Joint-continuous IK controller.
    Sends motion to real robot if REAL_ROBOT_MODE is True.
    """
    current_joints = robot.Joints().list()
    robot_dof = len(current_joints)

    all_solutions_mat = robot.SolveIK_All(
        target_pose,
        tool=tool_pose,
        reference=eye(4),
    )

    if all_solutions_mat.size(1) == 0:
        return False, "IK FAILED - NO SOLUTION"

    all_solutions = get_joint_solutions_from_mat(all_solutions_mat)
    valid_solutions = [s[:robot_dof] for s in all_solutions if len(s) >= robot_dof]

    if not valid_solutions:
        return False, "IK FAILED - NO VALID SOLUTION"

    best_solution = min(
        valid_solutions,
        key=lambda sol: joint_distance(sol, current_joints)
    )

    max_joint_change = max(
        abs(wrapped_joint_difference(best, current))
        for best, current in zip(best_solution, current_joints)
    )

    if max_joint_change > MAX_JOINT_STEP_DEG:
        return False, f"JOINT JUMP REJECTED: {max_joint_change:.1f} DEG"

    new_joints = limit_joint_step(
        current_joints,
        best_solution,
        max_step_deg=MAX_JOINT_STEP_DEG,
    )

    if REAL_ROBOT_MODE:
        robot.MoveJ(new_joints)
    else:
        robot.setJoints(new_joints)

    return True, "ROBOT FOLLOWING HAND"


def set_contactile_opening(gripper, opening):
    """Set simulation gripper joint position."""
    if gripper is None:
        return
    opening = float(np.clip(opening, 0.0, 1.0))
    joint_value = (
        GRIPPER_CLOSED_VALUE
        + opening * (GRIPPER_OPEN_VALUE - GRIPPER_CLOSED_VALUE)
    )
    gripper.setJoints([joint_value])


def check_gripper_object_contact(rdk, left_contact_item, right_contact_item, pick_object):
    """Collision check for Contactile fingers in simulation."""
    if pick_object is None or not pick_object.Valid():
        return False, False, False
    if left_contact_item is None or not left_contact_item.Valid():
        return False, False, False
    if right_contact_item is None or not right_contact_item.Valid():
        return False, False, False

    left_contact = (rdk.Collision(left_contact_item, pick_object) > 0)
    right_contact = (rdk.Collision(right_contact_item, pick_object) > 0)
    return (left_contact or right_contact), left_contact, right_contact


def update_contactile_animation(
    rdk, gripper, left_contact_item, right_contact_item,
    pick_object, current_opening, target_opening, contact_latched
):
    """Animate simulated Contactile gripper opening/closing."""
    left_contact = False
    right_contact = False

    if gripper is None:
        return current_opening, contact_latched, left_contact, right_contact

    difference = target_opening - current_opening

    if difference > 0.0:
        contact_latched = False
        new_opening = target_opening if abs(difference) < GRIPPER_STEP else current_opening + GRIPPER_STEP
        new_opening = float(np.clip(new_opening, 0.0, 1.0))
        set_contactile_opening(gripper, new_opening)
        return new_opening, contact_latched, left_contact, right_contact

    if contact_latched:
        return current_opening, True, left_contact, right_contact

    if current_opening <= target_opening:
        set_contactile_opening(gripper, target_opening)
        return target_opening, False, left_contact, right_contact

    previous_safe_opening = current_opening
    test_opening = max(target_opening, current_opening - CONTACT_CLOSE_STEP)
    set_contactile_opening(gripper, test_opening)
    time.sleep(CONTACT_SETTLE_SEC)

    contact_detected, left_contact, right_contact = check_gripper_object_contact(
        rdk, left_contact_item, right_contact_item, pick_object
    )

    if contact_detected:
        set_contactile_opening(gripper, previous_safe_opening)
        time.sleep(CONTACT_SETTLE_SEC)
        return previous_safe_opening, True, left_contact, right_contact

    return test_opening, False, left_contact, right_contact


def smooth_and_limit_pose(previous_pose, desired_pose):
    """Exponential smoothing and maximum step limiting for TCP target."""
    previous_position = np.array(previous_pose.Pos(), dtype=np.float64)
    desired_position = np.array(desired_pose.Pos(), dtype=np.float64)

    smoothed_position = (
        previous_position
        + TARGET_SMOOTHING_ALPHA * (desired_position - previous_position)
    )

    movement_vector = smoothed_position - previous_position
    movement_distance = np.linalg.norm(movement_vector)

    if movement_distance < MIN_TCP_STEP_MM:
        return deepcopy(previous_pose)

    if movement_distance > MAX_TCP_STEP_MM:
        movement_vector = (movement_vector / movement_distance) * MAX_TCP_STEP_MM
        smoothed_position = previous_position + movement_vector

    new_pose = deepcopy(previous_pose)
    new_pose.setPos(smoothed_position.tolist())
    return new_pose


def enforce_safe_workspace(desired_pose, workspace_center):
    """Enforce workspace box boundaries on requested target TCP."""
    desired_position = np.array(desired_pose.Pos(), dtype=np.float64)
    center = np.array(workspace_center, dtype=np.float64)

    min_position = np.array([
        center[0] - WORKSPACE_X_NEG_MM,
        center[1] - WORKSPACE_Y_NEG_MM,
        center[2] - WORKSPACE_Z_NEG_MM,
    ])
    max_position = np.array([
        center[0] + WORKSPACE_X_POS_MM,
        center[1] + WORKSPACE_Y_POS_MM,
        center[2] + WORKSPACE_Z_POS_MM,
    ])

    inside_workspace = np.all(desired_position >= min_position) and np.all(desired_position <= max_position)
    if inside_workspace:
        return desired_pose, False

    if not CLAMP_TO_WORKSPACE:
        return None, True

    limited_position = np.clip(desired_position, min_position, max_position)
    safe_pose = deepcopy(desired_pose)
    safe_pose.setPos(limited_position.tolist())
    return safe_pose, True


def detect_gripper_gesture(landmarks, pinch_active):
    """Detect Open Palm (OPEN) or Closed Fist (CLOSE) for gripper control."""
    if pinch_active:
        return None
    palm_center = calculate_palm_center(landmarks)
    ext_count = sum([
        is_finger_extended(landmarks, 8, 6, palm_center),
        is_finger_extended(landmarks, 12, 10, palm_center),
        is_finger_extended(landmarks, 16, 14, palm_center),
        is_finger_extended(landmarks, 20, 18, palm_center),
    ])
    if ext_count >= 4:
        return "OPEN"
    if ext_count <= 1:
        return "CLOSE"
    return None


def update_gripper_pick_drop(
    grip_attach_frame, base_frame, pick_object,
    gripper_state, gripper_current_opening, gripper_contact_detected, held_object
):
    """Simulation pick and drop logic."""
    if grip_attach_frame is None or not grip_attach_frame.Valid():
        return held_object
    if pick_object is None or not pick_object.Valid():
        return held_object

    if gripper_state == "OPEN" and held_object is not None:
        held_object.setParentStatic(base_frame)
        print(f"Released object: {held_object.Name()}")
        return None

    if gripper_state == "CLOSE" and held_object is None and gripper_contact_detected:
        grip_pos = np.array(grip_attach_frame.PoseAbs().Pos(), dtype=np.float64)
        obj_pos = np.array(pick_object.PoseAbs().Pos(), dtype=np.float64)
        dist_mm = float(np.linalg.norm(obj_pos - grip_pos))

        if dist_mm <= ATTACH_DISTANCE_MM:
            pick_object.setParentStatic(grip_attach_frame)
            pick_object.setPose(transl(*CUBE_HOLD_OFFSET_MM))
            print(f"Grabbed object: {pick_object.Name()}")
            return pick_object

    return held_object


def pose_distance_mm(pose_a, pose_b):
    """Distance between two TCP poses (mm)."""
    pos_a = np.array(pose_a.Pos(), dtype=np.float64)
    pos_b = np.array(pose_b.Pos(), dtype=np.float64)
    return float(np.linalg.norm(pos_a - pos_b))


def add_record_sample(recorded_path, recording_start_time, robot, tcp_pose, gripper_state, held_object):
    """Record single sample of robot joints and pose."""
    now = time.perf_counter()
    sample = {
        "time": now - recording_start_time,
        "joints": robot.Joints().list(),
        "pose": deepcopy(tcp_pose),
        "gripper_state": gripper_state,
        "holding": held_object is not None,
    }
    recorded_path.append(sample)
    return now, deepcopy(tcp_pose)


def save_recorded_path_csv(recorded_path):
    """Save path to CSV file."""
    RECORDINGS_FOLDER.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RECORDINGS_FOLDER / f"lara8_hand_recording_{timestamp}.csv"

    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "time_sec", "j1", "j2", "j3", "j4", "j5", "j6",
            "tcp_x", "tcp_y", "tcp_z", "gripper_state", "holding",
        ])
        for sample in recorded_path:
            j = sample["joints"]
            p = sample["pose"].Pos()
            writer.writerow([
                round(sample["time"], 4),
                *[round(v, 4) for v in j],
                round(p[0], 4), round(p[1], 4), round(p[2], 4),
                sample["gripper_state"], sample["holding"],
            ])
    print(f"Saved recording CSV: {csv_path}")
    return csv_path


def create_robodk_program_from_recording(rdk, robot, base_frame, recorded_path):
    """Create RoboDK program from recorded path."""
    if not recorded_path:
        return None
    timestamp = datetime.now().strftime("%H%M%S")
    program_name = f"{RECORDED_PROGRAM_NAME}_{timestamp}"
    program = rdk.AddProgram(program_name, robot)

    for idx, sample in enumerate(recorded_path):
        target = rdk.AddTarget(f"{program_name}_T{idx:03d}", base_frame, robot)
        target.setPose(sample["pose"])
        target.setJoints(sample["joints"])
        program.MoveJ(target)

    print(f'RoboDK program "{program_name}" created with {len(recorded_path)} points.')
    return program


def save_manual_waypoint(rdk, robot, base_frame, taught_waypoints, tcp_pose, gripper_state):
    """Save manual waypoint to RoboDK station and list."""
    idx = len(taught_waypoints) + 1
    target_name = f"{TEACH_TARGET_PREFIX}_{idx:03d}"
    target = rdk.AddTarget(target_name, base_frame, robot)
    target.setAsCartesianTarget()
    target.setPose(tcp_pose)
    target.setJoints(robot.Joints())

    waypoint = {
        "target": target,
        "pose": deepcopy(tcp_pose),
        "joints": robot.Joints().list(),
        "gripper_state": gripper_state,
    }
    taught_waypoints.append(waypoint)
    print(f"Saved manual waypoint {idx}: {target_name}")
    return taught_waypoints


def create_program_from_manual_waypoints(rdk, robot, taught_waypoints):
    """Create RoboDK program from taught waypoints."""
    if not taught_waypoints:
        print("No taught waypoints available.")
        return None
    timestamp = datetime.now().strftime("%H%M%S")
    program_name = f"{TEACH_PROGRAM_NAME}_{timestamp}"
    program = rdk.AddProgram(program_name, robot)

    for wp in taught_waypoints:
        program.MoveJ(wp["target"])

    print(f'Taught program "{program_name}" created with {len(taught_waypoints)} targets.')
    return program


def detect_save_waypoint_gesture(landmarks, pinch_active):
    """Detect V-sign gesture for saving waypoint."""
    if pinch_active:
        return False

    palm_center = calculate_palm_center(landmarks)
    index_ext = is_finger_extended(landmarks, 8, 6, palm_center)
    middle_ext = is_finger_extended(landmarks, 12, 10, palm_center)
    ring_ext = is_finger_extended(landmarks, 16, 14, palm_center)
    little_ext = is_finger_extended(landmarks, 20, 18, palm_center)

    palm_width = distance_2d(landmarks[5], landmarks[17])
    im_dist = distance_2d(landmarks[8], landmarks[12])
    separated = (im_dist > palm_width * V_SIGN_SEPARATION_RATIO)

    return index_ext and middle_ext and not ring_ext and not little_ext and separated


def is_new_waypoint_far_enough(taught_waypoints, tcp_pose):
    """Verify minimum distance to avoid duplicate waypoints."""
    if not taught_waypoints:
        return True
    last_pose = taught_waypoints[-1]["pose"]
    return pose_distance_mm(last_pose, tcp_pose) >= SAVE_TARGET_MIN_DISTANCE_MM


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model was not found: {MODEL_PATH.resolve()}"
        )

    # Connect to RoboDK
    rdk, robot, target, base_frame = connect_to_robodk()
    gripper = load_contactile_gripper(rdk)

    pick_object = rdk.Item(PICK_OBJECT_NAME)
    if not pick_object.Valid():
        print(f'Warning: pick object "{PICK_OBJECT_NAME}" was not found.')
        pick_object = None

    left_contact_item = rdk.Item(CONTACT_LEFT_ITEM_NAME)
    right_contact_item = rdk.Item(CONTACT_RIGHT_ITEM_NAME)
    grip_attach_frame = rdk.Item(GRIP_ATTACH_FRAME_NAME, ITEM_TYPE_FRAME)

    tool_pose = robot.PoseTool()
    starting_robot_joints = robot.Joints().list()

    lower_lim, upper_lim, *_ = robot.JointLimits()
    joint_lower_limits = lower_lim.list() if hasattr(lower_lim, "list") else list(lower_lim)
    joint_upper_limits = upper_lim.list() if hasattr(upper_lim, "list") else list(upper_lim)

    starting_tcp_pose = robot.SolveFK(
        starting_robot_joints,
        tool=tool_pose,
        reference=eye(4),
    )

    workspace_center = starting_tcp_pose.Pos()
    print("Safe workspace center:", [round(v, 2) for v in workspace_center])

    target.setPose(starting_tcp_pose)
    original_target_pose = starting_tcp_pose
    last_safe_target_pose = starting_tcp_pose
    last_robot_update_time = 0.0

    print("Connected to RoboDK.")
    print(f'Real Robot Mode: {"ENABLED" if REAL_ROBOT_MODE else "SIMULATION ONLY (Disabled)"}')

    # Initialize webcam
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open webcam index {CAMERA_INDEX}.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Configure MediaPipe Landmarker
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.60,
        min_hand_presence_confidence=0.60,
        min_tracking_confidence=0.60,
    )

    # State variables
    filtered_palm = None
    pinch_previous = False
    hand_anchor = None
    target_anchor_position = None
    hand_size_anchor = None
    filtered_hand_size = None

    gripper_candidate = None
    gripper_candidate_count = 0
    gripper_state = "OPEN"
    gripper_target_opening = 1.0
    gripper_current_opening = 1.0
    gripper_contact_latched = False
    left_contact = False
    right_contact = False

    axis_mode = AXIS_MODE_DEFAULT
    axis_candidate = None
    axis_candidate_count = 0
    axis_hand_present = False

    control_mode = CONTROL_MODE_DEFAULT
    mode_gesture_start_time = None
    mode_gesture_candidate = None
    mode_gesture_status = "READY"
    cartesian_rearm_required = False

    selected_joint = None
    joint_candidate = None
    joint_candidate_count = 0
    joint_hand_anchor_x = None
    joint_angle_anchor = None
    last_joint_update_time = 0.0

    start_time = time.perf_counter()
    previous_timestamp_ms = -1
    held_object = None

    recording_active = False
    recorded_path = []
    recording_start_time = 0.0
    last_record_time = 0.0
    last_record_pose = None
    taught_waypoints = []

    save_gesture_start_time = None
    save_gesture_latched = False
    save_gesture_detected = False
    save_gesture_status = "OFF"

    print("\nLARA8 Real-Robot-Ready 2-Hand Controller active.")
    print("Controls:")
    print("  RIGHT HAND pinch: move Hand_Target (Cartesian) / move Joint (Joint mode)")
    print("  LEFT HAND fist: pause movement")
    print("  LEFT HAND 1 finger: X-axis only")
    print("  LEFT HAND 2 fingers: Y-axis only")
    print("  LEFT HAND 3 fingers: Z-axis only")
    print("  LEFT HAND 4 fingers: free XYZ movement")
    print("  LEFT HAND Thumbs Up/Down: Toggle CARTESIAN / JOINT mode")
    print("  RIGHT HAND V-sign (hold): Save waypoint")
    print("  RIGHT HAND Open Palm / Fist: Gripper OPEN / CLOSE")
    print("  R: Reset robot & target")
    print("  T: Teach manual waypoint")
    print("  G: Generate RoboDK program from taught waypoints")
    print("  X: Clear waypoints list")
    print("  Q or Esc: Exit\n")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Warning: webcam frame could not be read.")
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                media_pipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = int((time.perf_counter() - start_time) * 1000)
                if timestamp_ms <= previous_timestamp_ms:
                    timestamp_ms = previous_timestamp_ms + 1
                previous_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(media_pipe_image, timestamp_ms)

                tracking_status = "NO RIGHT HAND"
                status_color = (0, 0, 255)
                pinch_active = False
                pinch_ratio = 0.0
                palm_pixel = None
                save_gesture_detected = False
                save_gesture_status = "OFF"
                axis_hand_present = False

                hands_by_label = get_hands_by_label(result)

                # Draw detected hands
                if CONTROL_HAND_LABEL in hands_by_label:
                    draw_hand(frame, hands_by_label[CONTROL_HAND_LABEL], CONTROL_HAND_LABEL, (0, 255, 0))
                if AXIS_HAND_LABEL in hands_by_label:
                    draw_hand(frame, hands_by_label[AXIS_HAND_LABEL], AXIS_HAND_LABEL, (255, 165, 0))

                # ----------------------------------------------------
                # Left Hand: Mode selection & Axis locking / Joint selection
                # ----------------------------------------------------
                axis_landmarks = hands_by_label.get(AXIS_HAND_LABEL)
                control_landmarks = hands_by_label.get(CONTROL_HAND_LABEL)

                if axis_landmarks is not None:
                    axis_hand_present = True
                    detected_mode_gesture = detect_control_mode_gesture(axis_landmarks)

                    if detected_mode_gesture is not None:
                        current_mode_time = time.perf_counter()
                        if detected_mode_gesture == mode_gesture_candidate:
                            if mode_gesture_start_time is None:
                                mode_gesture_start_time = current_mode_time
                            hold_duration = current_mode_time - mode_gesture_start_time
                            mode_gesture_status = f"HOLD {detected_mode_gesture} {hold_duration:.1f}s"

                            if hold_duration >= MODE_SWITCH_HOLD_SEC:
                                if detected_mode_gesture != control_mode:
                                    control_mode = detected_mode_gesture
                                    print(f"Control mode changed to: {control_mode}")
                                    pinch_previous = False
                                    hand_anchor = None
                                    target_anchor_position = None
                                    hand_size_anchor = None
                                    joint_hand_anchor_x = None
                                    joint_angle_anchor = None

                                    if control_mode == "CARTESIAN":
                                        cartesian_rearm_required = True

                                mode_gesture_status = f"MODE: {control_mode}"
                        else:
                            mode_gesture_candidate = detected_mode_gesture
                            mode_gesture_start_time = current_mode_time
                    else:
                        mode_gesture_candidate = None
                        mode_gesture_start_time = None
                        mode_gesture_status = "READY"

                    if control_mode == "CARTESIAN":
                        detected_axis = detect_axis_mode(axis_landmarks)
                        if detected_axis is not None:
                            if detected_axis == axis_candidate:
                                axis_candidate_count += 1
                            else:
                                axis_candidate = detected_axis
                                axis_candidate_count = 1

                            if axis_candidate_count >= AXIS_DEBOUNCE_FRAMES and detected_axis != axis_mode:
                                axis_mode = detected_axis
                                print(f"Selected axis mode: {axis_mode}")
                                pinch_previous = False
                                hand_anchor = None
                                target_anchor_position = None
                                hand_size_anchor = None

                    elif control_mode == "JOINT":
                        detected_joint = detect_joint_selection_gesture(axis_landmarks)
                        if detected_joint is not None:
                            if detected_joint == joint_candidate:
                                joint_candidate_count += 1
                            else:
                                joint_candidate = detected_joint
                                joint_candidate_count = 1

                            if joint_candidate_count >= JOINT_SELECT_DEBOUNCE_FRAMES and detected_joint != selected_joint:
                                selected_joint = detected_joint
                                print(f"Selected robot joint: J{selected_joint}")
                                joint_hand_anchor_x = None
                                joint_angle_anchor = None
                                pinch_previous = False

                else:
                    axis_candidate = None
                    axis_candidate_count = 0
                    if REQUIRE_AXIS_HAND:
                        pinch_previous = False
                        hand_anchor = None
                        target_anchor_position = None
                        hand_size_anchor = None

                # ----------------------------------------------------
                # Right Hand: Movement & Gripper & Waypoint Save
                # ----------------------------------------------------
                if control_landmarks is not None:
                    landmarks = control_landmarks
                    raw_palm = calculate_palm_center(landmarks)

                    if filtered_palm is None:
                        filtered_palm = raw_palm.copy()
                    else:
                        filtered_palm = SMOOTHING_ALPHA * raw_palm + (1.0 - SMOOTHING_ALPHA) * filtered_palm

                    palm_width = distance_2d(landmarks[5], landmarks[17])
                    if filtered_hand_size is None:
                        filtered_hand_size = palm_width
                    else:
                        filtered_hand_size = DEPTH_SMOOTHING_ALPHA * palm_width + (1.0 - DEPTH_SMOOTHING_ALPHA) * filtered_hand_size

                    thumb_index_distance = distance_2d(landmarks[4], landmarks[8])
                    index_length = distance_2d(landmarks[5], landmarks[8])

                    pinch_ratio = thumb_index_distance / max(index_length, 0.0001)
                    pinch_active = (pinch_ratio < PINCH_THRESHOLD)

                    gripper_gesture = None
                    if ENABLE_GESTURE_WAYPOINT_SAVE:
                        save_gesture_detected = detect_save_waypoint_gesture(landmarks, pinch_active)

                    if ENABLE_GRIPPER_GESTURES and not save_gesture_detected:
                        gripper_gesture = detect_gripper_gesture(landmarks, pinch_active)

                    if gripper_gesture is None:
                        gripper_candidate = None
                        gripper_candidate_count = 0
                    else:
                        if gripper_gesture == gripper_candidate:
                            gripper_candidate_count += 1
                        else:
                            gripper_candidate = gripper_gesture
                            gripper_candidate_count = 1

                        if gripper_candidate_count >= GRIPPER_DEBOUNCE_FRAMES and gripper_gesture != gripper_state:
                            gripper_state = gripper_gesture
                            if gripper_state == "OPEN":
                                gripper_target_opening = 1.0
                                send_real_gripper_command(1.0)
                                print("Contactile gripper command: OPEN")
                            elif gripper_state == "CLOSE":
                                gripper_target_opening = 0.0
                                send_real_gripper_command(0.0)
                                print("Contactile gripper command: CLOSE")

                    height, width = frame.shape[:2]
                    palm_pixel = (int(filtered_palm[0] * width), int(filtered_palm[1] * height))

                    # --- CARTESIAN MODE ---
                    if control_mode == "CARTESIAN":
                        if cartesian_rearm_required and not pinch_active:
                            cartesian_rearm_required = False
                            print("Cartesian control ready. Select axis and pinch to move.")

                        movement_allowed = (
                            not cartesian_rearm_required
                            and pinch_active
                            and axis_mode != "PAUSE"
                            and (axis_hand_present or not REQUIRE_AXIS_HAND)
                        )

                        if pinch_active and not movement_allowed:
                            if cartesian_rearm_required:
                                tracking_status = "RELEASE RIGHT PINCH TO ARM CARTESIAN"
                            elif axis_mode == "PAUSE":
                                tracking_status = "PINCH ACTIVE - LEFT HAND PAUSED"
                            else:
                                tracking_status = "PINCH ACTIVE - SHOW LEFT AXIS HAND"

                            status_color = (0, 165, 255)
                            pinch_previous = False
                            hand_anchor = None
                            target_anchor_position = None
                            hand_size_anchor = None

                        elif movement_allowed:
                            tracking_status = f"PINCH ACTIVE - {axis_mode} MODE"
                            status_color = (0, 255, 0)

                            if not pinch_previous or (axis_candidate_count == 1):
                                hand_anchor = filtered_palm.copy()
                                hand_size_anchor = filtered_hand_size
                                anchor_pose = target.Pose()
                                target_anchor_position = anchor_pose.Pos()

                            if hand_anchor is not None and target_anchor_position is not None:
                                hand_delta = filtered_palm - hand_anchor
                                horizontal_delta = hand_delta[0]
                                vertical_delta = hand_delta[1]

                                if abs(horizontal_delta) < DEAD_ZONE:
                                    horizontal_delta = 0.0
                                if abs(vertical_delta) < DEAD_ZONE:
                                    vertical_delta = 0.0

                                robot_dy = horizontal_delta * X_SCALE_MM
                                if vertical_delta < 0:
                                    robot_dz = -vertical_delta * Z_UP_SCALE_MM
                                else:
                                    robot_dz = -vertical_delta * Z_DOWN_SCALE_MM

                                robot_dy = float(np.clip(robot_dy, -MAX_X_MOVEMENT_MM, MAX_X_MOVEMENT_MM))
                                robot_dz = float(np.clip(robot_dz, -MAX_Z_MOVEMENT_MM, MAX_Z_MOVEMENT_MM))

                                robot_dx = 0.0
                                if ENABLE_WEBCAM_DEPTH and hand_size_anchor is not None:
                                    size_ratio = filtered_hand_size / max(hand_size_anchor, 0.0001)
                                    robot_dx = (size_ratio - 1.0) * Y_SCALE_MM * Y_DIRECTION
                                    robot_dx = float(np.clip(robot_dx, -MAX_Y_MOVEMENT_MM, MAX_Y_MOVEMENT_MM))

                                robot_dx, robot_dy, robot_dz = apply_axis_lock(
                                    robot_dx, robot_dy, robot_dz, axis_mode
                                )

                                command_position = [
                                    target_anchor_position[0] + robot_dx,
                                    target_anchor_position[1] + robot_dy,
                                    target_anchor_position[2] + robot_dz,
                                ]

                                command_pose = target.Pose()
                                command_pose.setPos(command_position)

                                current_time = time.perf_counter()
                                if current_time - last_robot_update_time >= ROBOT_UPDATE_INTERVAL:
                                    safe_command_pose, workspace_limited = enforce_safe_workspace(
                                        command_pose, workspace_center
                                    )

                                    if safe_command_pose is None:
                                        accepted = False
                                        robot_status = "OUTSIDE SAFE WORKSPACE"
                                        smooth_pose = last_safe_target_pose
                                    else:
                                        smooth_pose = smooth_and_limit_pose(last_safe_target_pose, safe_command_pose)
                                        accepted, robot_status = move_robot_to_target(robot, smooth_pose, tool_pose)
                                        if workspace_limited and accepted:
                                            robot_status = "WORKSPACE LIMIT ACTIVE"

                                    last_robot_update_time = current_time

                                    if accepted:
                                        target.setPose(smooth_pose)
                                        last_safe_target_pose = smooth_pose
                                        status_color = (0, 255, 0)
                                    else:
                                        target.setPose(last_safe_target_pose)
                                        status_color = (0, 0, 255)
                                        tracking_status = robot_status

                                cv2.putText(
                                    frame,
                                    f"Applied {axis_mode}: X={robot_dx:+.1f} Y={robot_dy:+.1f} Z={robot_dz:+.1f} mm",
                                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                                )

                            pinch_previous = True

                        else:
                            tracking_status = "RIGHT HAND FOUND - PINCH TO MOVE"
                            status_color = (0, 255, 255)
                            pinch_previous = False

                    # --- JOINT MODE ---
                    elif control_mode == "JOINT":
                        if pinch_active and selected_joint is not None:
                            tracking_status = f"JOINT MODE - MOVING J{selected_joint}"
                            status_color = (0, 165, 255)

                            if not pinch_previous or joint_hand_anchor_x is None or joint_angle_anchor is None:
                                joint_hand_anchor_x = filtered_palm[0]
                                current_joints = robot.Joints().list()
                                joint_angle_anchor = current_joints[selected_joint - 1]

                            horizontal_delta = filtered_palm[0] - joint_hand_anchor_x
                            if abs(horizontal_delta) < DEAD_ZONE:
                                horizontal_delta = 0.0

                            requested_joint_change = horizontal_delta * JOINT_HAND_SCALE_DEG
                            requested_joint_change = float(np.clip(
                                requested_joint_change, -JOINT_MAX_FROM_ANCHOR_DEG, JOINT_MAX_FROM_ANCHOR_DEG
                            ))

                            desired_joint_angle = joint_angle_anchor + requested_joint_change
                            joint_index = selected_joint - 1
                            desired_joint_angle = float(np.clip(
                                desired_joint_angle, joint_lower_limits[joint_index], joint_upper_limits[joint_index]
                            ))

                            current_time = time.perf_counter()
                            if current_time - last_joint_update_time >= JOINT_UPDATE_INTERVAL:
                                current_joints = robot.Joints().list()
                                current_angle = current_joints[joint_index]
                                angle_difference = desired_joint_angle - current_angle
                                angle_step = float(np.clip(angle_difference, -JOINT_CONTROL_STEP_DEG, JOINT_CONTROL_STEP_DEG))

                                current_joints[joint_index] = current_angle + angle_step

                                if REAL_ROBOT_MODE:
                                    robot.MoveJ(current_joints)
                                else:
                                    robot.setJoints(current_joints)

                                current_tcp_pose = robot.SolveFK(current_joints, tool=tool_pose, reference=eye(4))
                                target.setPose(current_tcp_pose)
                                last_safe_target_pose = deepcopy(current_tcp_pose)
                                last_joint_update_time = current_time

                            pinch_previous = True

                        elif pinch_active and selected_joint is None:
                            tracking_status = "JOINT MODE - SELECT A JOINT (J1-J6)"
                            status_color = (0, 165, 255)
                            pinch_previous = False

                    if control_mode == "JOINT" and not pinch_active:
                        joint_hand_anchor_x = None
                        joint_angle_anchor = None
                        if selected_joint is not None:
                            tracking_status = f"JOINT MODE - J{selected_joint} SELECTED"
                        else:
                            tracking_status = "JOINT MODE - SELECT A JOINT"
                        status_color = (0, 255, 255)
                        pinch_previous = False

                    cv2.putText(
                        frame, f"Right pinch ratio: {pinch_ratio:.2f}",
                        (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )

                else:
                    pinch_previous = False
                    hand_anchor = None
                    target_anchor_position = None
                    filtered_palm = None
                    filtered_hand_size = None
                    hand_size_anchor = None
                    gripper_candidate = None
                    gripper_candidate_count = 0
                    joint_hand_anchor_x = None
                    joint_angle_anchor = None

                cv2.putText(
                    frame, tracking_status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2
                )

                if palm_pixel is not None:
                    cv2.circle(frame, palm_pixel, 12, (0, 255, 0), -1)

                # V-sign Waypoint Saving
                if ENABLE_GESTURE_WAYPOINT_SAVE:
                    current_gesture_time = time.perf_counter()
                    if save_gesture_detected:
                        if save_gesture_latched:
                            save_gesture_status = "SAVED - RELEASE V SIGN"
                        else:
                            if save_gesture_start_time is None:
                                save_gesture_start_time = current_gesture_time
                            hold_time = current_gesture_time - save_gesture_start_time
                            save_gesture_status = f"HOLD V SIGN {hold_time:.1f}s"

                            if hold_time >= SAVE_GESTURE_HOLD_SEC:
                                if is_new_waypoint_far_enough(taught_waypoints, last_safe_target_pose):
                                    taught_waypoints = save_manual_waypoint(
                                        rdk, robot, base_frame, taught_waypoints, last_safe_target_pose, gripper_state
                                    )
                                    save_gesture_status = "TARGET SAVED"
                                else:
                                    save_gesture_status = "TOO CLOSE TO LAST TARGET"
                                save_gesture_latched = True
                    else:
                        save_gesture_start_time = None
                        save_gesture_latched = False
                        save_gesture_detected = False
                        save_gesture_status = "READY"

                # Update simulation gripper & pick/drop
                (gripper_current_opening, gripper_contact_latched, left_contact, right_contact) = update_contactile_animation(
                    rdk, gripper, left_contact_item, right_contact_item, pick_object,
                    gripper_current_opening, gripper_target_opening, gripper_contact_latched
                )

                if ENABLE_PICK_AND_DROP:
                    held_object = update_gripper_pick_drop(
                        grip_attach_frame, base_frame, pick_object,
                        gripper_state, gripper_current_opening, gripper_contact_latched, held_object
                    )

                # Display On-Screen Overlays
                cv2.putText(
                    frame, f"Axis mode: {axis_mode} | Left hand: {'FOUND' if axis_hand_present else 'MISSING'}",
                    (20, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                )
                cv2.putText(
                    frame, f"Gripper: {gripper_state} | Real Robot: {'ON' if REAL_ROBOT_MODE else 'OFF (Sim)'}",
                    (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
                )
                cv2.putText(
                    frame, f"Taught targets: {len(taught_waypoints)} | V-Save: {save_gesture_status}",
                    (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2
                )
                cv2.putText(
                    frame, f"CONTROL MODE: {control_mode} | {mode_gesture_status}",
                    (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0) if control_mode == "CARTESIAN" else (0, 165, 255), 2
                )

                if control_mode == "JOINT":
                    if selected_joint is None:
                        joint_text = "Selected Joint: NONE"
                    else:
                        current_j_vals = robot.Joints().list()
                        joint_text = f"Selected Joint: J{selected_joint} | {current_j_vals[selected_joint - 1]:.1f} deg"
                    cv2.putText(frame, joint_text, (20, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

                cv2.putText(
                    frame, "Right pinch: move | Left hand: axis/mode | R: reset | T: teach | Q: exit",
                    (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )

                cv2.imshow("LARA8 Real-Robot Test - 2-Hand Axis Lock Controller", cv2.resize(frame, (640, 360)))

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

                if key == ord("r"):
                    if REAL_ROBOT_MODE:
                        robot.MoveJ(starting_robot_joints)
                    else:
                        robot.setJoints(starting_robot_joints)

                    target.setPose(original_target_pose)
                    last_safe_target_pose = original_target_pose
                    pinch_previous = False
                    hand_anchor = None
                    target_anchor_position = None
                    filtered_palm = None
                    filtered_hand_size = None
                    hand_size_anchor = None
                    axis_mode = AXIS_MODE_DEFAULT
                    axis_candidate = None
                    axis_candidate_count = 0
                    control_mode = CONTROL_MODE_DEFAULT
                    mode_gesture_start_time = None
                    mode_gesture_candidate = None
                    mode_gesture_status = "READY"
                    cartesian_rearm_required = False
                    print("Robot and Hand_Target reset.")

                if key == ord("t") and ENABLE_MANUAL_WAYPOINT_TEACHING:
                    taught_waypoints = save_manual_waypoint(
                        rdk, robot, base_frame, taught_waypoints, last_safe_target_pose, gripper_state
                    )

                if key == ord("g") and ENABLE_MANUAL_WAYPOINT_TEACHING:
                    create_program_from_manual_waypoints(rdk, robot, taught_waypoints)

                if key == ord("x"):
                    taught_waypoints.clear()
                    print("Cleared taught waypoints list.")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Real-robot 2-hand controller stopped.")


if __name__ == "__main__":
    main()
