"""
uFactory 850 & uFactory xArm Gripper - Real Robot Test Ready Version

Features:
    - 2-Hand Gesture Control:
        * Right hand: Pinch to move robot TCP (or control joint in Joint Mode)
        * Right hand: Closed fist held to toggle gripper (OPEN / CLOSE)
        * Right hand: V-sign held for 0.5s to teach/save manual waypoints
        * Left hand: Axis lock selection (Fist=PAUSE, 1 finger=X, 2 fingers=Y, 3 fingers=Z, Open palm=FREE)
        * Left hand: Joint mode selection (J1-J6 selection via finger gestures)
    - Mode Switching: Thumbs-up / Thumbs-down to toggle between CARTESIAN and JOINT modes
    - Safety Systems:
        * REAL_ROBOT_MODE toggle (Simulation default vs Live robot execution)
        * REAL_GRIPPER_COMMANDS_ENABLED toggle
        * Conservative speed and acceleration caps for physical testing
        * Safety re-arm: must release pinch before movement arms
        * IK Jump rejection and joint velocity stepping limits
        * Bounded safe 3D workspace clamping
        * Left-hand gesture axis locking for controlled single-axis verification
        * 'r' key emergency / manual return to initial home posture

REAL ROBOT TEST PROTOCOL:
    1. Verify robot connection in RoboDK:
       - Open RoboDK -> Right-click "uFactory 850" -> "Connect to robot"
       - Enter robot IP address -> Test connection -> Connect
       - Confirm RoboDK driver mode is connected.
    2. Keep physical uFactory 850 emergency stop in your hand at all times.
    3. Test in RoboDK simulation first (REAL_ROBOT_MODE = False).
    4. Once verified, set REAL_ROBOT_MODE = True.
    5. Start with single-axis locks on the left hand (e.g. 1 finger for X only)
       before using Free 3D motion.
"""
from collections import abc
from robodk import robomath
from robodk import robomath
from robodk import robomath
from robodk import robomath
from robodk import robomath
import math
import time
from pathlib import Path
from copy import deepcopy  #smoothwere need to be imported from deepcopy module

import csv
from datetime import datetime #for path recording

import cv2
import mediapipe as mp
import numpy as np

from robodk.robolink import (
    Robolink,
    ITEM_TYPE_FRAME,
    ITEM_TYPE_TARGET,
    ITEM_TYPE_ROBOT,
    ITEM_TYPE_TOOL,
    INSTRUCTION_CALL_PROGRAM,
)

from robodk.robomath import eye, transl


# ============================================================
# CONFIGURATION
# ============================================================

# ============================================================
# REAL ROBOT TEST SWITCHES
# ============================================================

# Keep this False for simulation. Set True ONLY when the RoboDK station is
# connected to the real uFactory 850 robot through the RoboDK robot driver.
REAL_ROBOT_MODE = False

# Enable sending real gripper commands (via RoboDK program calls / driver)
REAL_GRIPPER_COMMANDS_ENABLED = True

# Conservative speeds and accelerations for initial real cobot testing
REAL_ROBOT_SPEED_MM_S = 40.0           # Linear speed (mm/s)
REAL_ROBOT_SPEED_JOINT_DEG_S = 15.0     # Joint speed (deg/s)
REAL_ROBOT_ACCEL_MM_S2 = 100.0         # Linear acceleration (mm/s²)
REAL_ROBOT_ACCEL_JOINT_DEG_S2 = 30.0   # Joint acceleration (deg/s²)




MODEL_PATH = Path("hand_landmarker.task")

CAMERA_INDEX = 0

TARGET_NAME = "Hand_Target"
BASE_FRAME_NAME = "uFactory 850 Base"

# Maximum target displacement from the point where pinch starts
MAX_X_MOVEMENT_MM = 250.0
MAX_Z_MOVEMENT_MM = 250.0

# Full normalized webcam width/height mapped to RoboDK movement
X_SCALE_MM = 750.0
Z_SCALE_MM = 400.0

# Lower value = smoother but slower response
SMOOTHING_ALPHA = 0.30

# ============================================================
# ROBUST PINCH DETECTION
# ============================================================

# Pinch must become clearly small before movement starts
PINCH_ENGAGE_THRESHOLD = 0.35

# Pinch must become clearly open before movement stops
# This is intentionally higher than ENGAGE threshold.
PINCH_RELEASE_THRESHOLD = 0.50

# Number of consecutive frames required
PINCH_ENGAGE_FRAMES = 3
PINCH_RELEASE_FRAMES = 5

# Minimum hand movement before updating the target
DEAD_ZONE = 0.004

ROBOT_NAME = "uFactory 850"

# Maximum joint movement allowed in one webcam update
MAX_JOINT_STEP_DEG = 20.0

# Limit robot updates to approximately 20 Hz
ROBOT_UPDATE_INTERVAL = 0.035

# Extra smoothing for the robot TCP motion
TARGET_SMOOTHING_ALPHA = 0.28

# Maximum TCP movement allowed per robot update in mm
MAX_TCP_STEP_MM = 8.0

# Ignore very tiny TCP movements
MIN_TCP_STEP_MM = 0.2

#downward movement
Z_UP_SCALE_MM = 450.0
Z_DOWN_SCALE_MM = 1200.0

# Safe workspace around the starting TCP position, in mm
WORKSPACE_X_NEG_MM = 350.0
WORKSPACE_X_POS_MM = 350.0

WORKSPACE_Y_NEG_MM = 250.0
WORKSPACE_Y_POS_MM = 250.0

WORKSPACE_Z_NEG_MM = 550.0
WORKSPACE_Z_POS_MM = 300.0

# True = clamp target inside workspace
# False = reject target outside workspace
CLAMP_TO_WORKSPACE = True


# Approximate webcam depth using palm size
ENABLE_WEBCAM_DEPTH = True

# How much robot Y movement is created from hand-size change
Y_SCALE_MM = 220.0

# Maximum Y movement from the pinch start position
MAX_Y_MOVEMENT_MM = 100.0

# Smoothing for estimated hand size/depth
DEPTH_SMOOTHING_ALPHA = 0.15

# If direction feels reversed, change this to -1.0
Y_DIRECTION = 1.0

# ============================================================
# PATH RECORDING
# ============================================================

ENABLE_PATH_RECORDING = False


RECORD_INTERVAL_SEC = 0.25
RECORD_MIN_TCP_DISTANCE_MM = 15.0
MAX_RECORDED_POINTS = 40

RECORDINGS_FOLDER = Path("recordings")
RECORDED_PROGRAM_NAME = "Hand_Guided_xArm850_Path"

# ============================================================
# MANUAL WAYPOINT TEACHING
# ============================================================

ENABLE_MANUAL_WAYPOINT_TEACHING = True

TEACH_TARGET_PREFIX = "Hand_Taught_Target"
TEACH_PROGRAM_NAME = "Hand_Taught_Program"

# ============================================================
# uFACTORY xARM GRIPPER CONTROL
# ============================================================

ENABLE_GRIPPER_GESTURES = True

# Exact tool name in RoboDK station tree
GRIPPER_NAME = "uFactory xArm Gripper"

# Opening limits:
# 1.0 = fully open
# 0.0 = fully closed
GRIPPER_OPEN_VALUE = 1.0
GRIPPER_CLOSED_VALUE = 0.0

# How many stable frames before accepting open/close gesture
GRIPPER_DEBOUNCE_FRAMES = 8

# ============================================================
# SAFE LATCHED GRIPPER CONTROL
# ============================================================

# Closing can be slightly faster
GRIPPER_CLOSE_HOLD_SEC = 0.70

# Opening/releasing requires a longer deliberate hold
GRIPPER_OPEN_HOLD_SEC = 1.20

# Number of non-fist frames required before another
# gripper command can be accepted
GRIPPER_RELEASE_FRAMES = 6

# Animation speed
GRIPPER_STEP = 0.08

PICK_OBJECT_NAME = "Sphere D100mm"

GRIP_ATTACH_FRAME_NAME = "Grip_Attach_Frame"

# Attachment distance so sphere attaches when close to the gripper jaws
ATTACH_DISTANCE_MM = 150.0

# Local object position after attaching to grip frame
CUBE_HOLD_OFFSET_MM = [0, 0, 0]


# ============================================================
# GESTURE-BASED WAYPOINT SAVING
# ============================================================

ENABLE_GESTURE_WAYPOINT_SAVE = True

# V-sign must be held for this time before saving one waypoint
SAVE_GESTURE_HOLD_SEC = 0.50

# Avoid duplicate targets at almost the same robot position
SAVE_TARGET_MIN_DISTANCE_MM = 20.0

# How separated index and middle fingers should be for a clear V sign
V_SIGN_SEPARATION_RATIO = 0.20


# ============================================================
# ADAPTIVE CONTACT-BASED CLOSING
# ============================================================

# Normalized closing step
CONTACT_CLOSE_STEP = 0.02

# Settle time between animation checks
CONTACT_SETTLE_SEC = 0.01

# ============================================================
# TWO-HAND AXIS-LOCK CONTROL
# ============================================================

# Right hand: robot movement, gripper gestures and waypoint saving
CONTROL_HAND_LABEL = "Right"

# Left hand: selects the Cartesian movement axis
AXIS_HAND_LABEL = "Left"

# Set True only if MediaPipe labels your hands in reverse
SWAP_HAND_LABELS = True

# Require the axis-selection hand to be visible before robot movement
REQUIRE_AXIS_HAND = True

# Initial mode before the left-hand gesture is accepted
AXIS_MODE_DEFAULT = "PAUSE"

# Stable frames required before changing axis mode
AXIS_DEBOUNCE_FRAMES = 6

# ============================================================
# CARTESIAN / JOINT MODE SELECTION
# ============================================================

CONTROL_MODE_DEFAULT = "CARTESIAN"

# Hold thumbs-up/down for this long before accepting it
MODE_SWITCH_HOLD_SEC = 0.50

# Minimum vertical separation used to identify thumb direction
THUMB_MODE_MARGIN = 0.04

# ============================================================
# JOINT CONTROL CONFIGURATION
# ============================================================

JOINT_SELECT_DEBOUNCE_FRAMES = 6
JOINT_HAND_SCALE_DEG = 180.0
JOINT_MAX_FROM_ANCHOR_DEG = 45.0
JOINT_CONTROL_STEP_DEG = 2.0
JOINT_UPDATE_INTERVAL = 0.035

# ============================================================
# DIRECT ON-SCREEN JOINT SELECTOR
# ============================================================

# Point at a joint box for this long to select it
JOINT_SELECTOR_HOLD_SEC = 0.50

# After selection, pointer must leave the selector before
# another joint can be selected
JOINT_SELECTOR_RELEASE_FRAMES = 6

# Selector area in normalized webcam coordinates
JOINT_SELECTOR_X_MIN = 0.08
JOINT_SELECTOR_X_MAX = 0.92

JOINT_SELECTOR_Y_MIN = 0.60
JOINT_SELECTOR_Y_MAX = 0.76

# ============================================================
# HAND CONNECTIONS FOR DRAWING
# ============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

def load_gripper(rdk):
    gripper = rdk.Item(GRIPPER_NAME)

    if not gripper.Valid():
        print(f'Warning: Gripper "{GRIPPER_NAME}" was not found.')
        return None

    print(f'Loaded gripper: "{gripper.Name()}" (Type {gripper.Type()})')
    return gripper

def distance_2d(point_a, point_b):
    """Return the 2D distance between two normalized landmarks."""
    return math.hypot(
        point_a.x - point_b.x,
        point_a.y - point_b.y,
    )


def calculate_palm_center(landmarks):
    """
    More stable palm centre using wrist, palm base, and knuckle points.
    Fingertips are avoided because they move a lot during gestures.
    """
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


def draw_hand(
    frame,
    landmarks,
    label="",
    line_color=(255, 255, 255),
):
    """Draw one detected hand and an optional role label."""
    height, width = frame.shape[:2]

    points = []

    for landmark in landmarks:
        pixel_x = int(landmark.x * width)
        pixel_y = int(landmark.y * height)
        points.append((pixel_x, pixel_y))

    for start_index, end_index in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start_index],
            points[end_index],
            line_color,
            2,
        )

    for index, point in enumerate(points):
        color = (0, 255, 255)

        if index == 4:
            color = (255, 0, 255)   # thumb tip

        if index == 8:
            color = (0, 255, 0)     # index tip

        cv2.circle(frame, point, 5, color, -1)

    if label and points:
        label_position = (
            max(10, points[0][0] - 35),
            max(25, points[0][1] + 30),
        )

        cv2.putText(
            frame,
            label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            line_color,
            2,
        )


def get_hands_by_label(result):
    """
    Return detected hands using MediaPipe handedness labels.
    """
    hands = {}

    if not result.hand_landmarks:
        return hands

    for landmarks, handedness in zip(
        result.hand_landmarks,
        result.handedness,
    ):
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


def count_extended_fingers(landmarks):
    """Count extended index, middle, ring and little fingers."""
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

    Index, middle, ring and little fingers should be folded.
    """

    # Avoid confusing normal axis-selection gestures
    # with the mode-selection gesture.
    if count_extended_fingers(landmarks) != 0:
        return None

    palm_center = calculate_palm_center(landmarks)

    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]

    # Make sure the thumb is actually extended away from the palm.
    thumb_tip_distance = math.hypot(
        thumb_tip.x - palm_center[0],
        thumb_tip.y - palm_center[1],
    )

    thumb_mcp_distance = math.hypot(
        thumb_mcp.x - palm_center[0],
        thumb_mcp.y - palm_center[1],
    )

    thumb_extended = (
        thumb_tip_distance
        > thumb_mcp_distance * 1.25
    )

    if not thumb_extended:
        return None

    # Image Y decreases upward.
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
    """
    Detect whether the thumb is clearly spread away from the palm.

    Used only to distinguish:
        J4 = four fingers + thumb folded
        J5 = full open palm
    """

    palm_width = distance_2d(
        landmarks[5],   # index MCP
        landmarks[17],  # little-finger MCP
    )

    if palm_width < 0.001:
        return False

    thumb_distance = distance_2d(
        landmarks[4],   # thumb tip
        landmarks[5],   # index MCP
    )

    thumb_ratio = thumb_distance / palm_width

    return thumb_ratio > 0.75

def detect_joint_selection_gesture(landmarks):
    """
    Left-hand joint-selection gestures:

        Index only                  -> J1
        Index + Middle              -> J2
        Index + Middle + Ring       -> J3
        Four fingers, thumb folded  -> J4
        Full open palm              -> J5
        Closed fist                 -> J6

    Thumb is ignored for J6 because fist thumb position can vary.
    """

    palm_center = calculate_palm_center(landmarks)

    index_extended = is_finger_extended(
        landmarks,
        8,
        6,
        palm_center,
    )

    middle_extended = is_finger_extended(
        landmarks,
        12,
        10,
        palm_center,
    )

    ring_extended = is_finger_extended(
        landmarks,
        16,
        14,
        palm_center,
    )

    little_extended = is_finger_extended(
        landmarks,
        20,
        18,
        palm_center,
    )

    thumb_extended = is_thumb_extended_for_joint(
        landmarks
    )

    # --------------------------------------------------------
    # J6 - closed fist
    # --------------------------------------------------------
    if (
        not index_extended
        and not middle_extended
        and not ring_extended
        and not little_extended
    ):
        return 6

    # --------------------------------------------------------
    # J1 - index only
    # --------------------------------------------------------
    if (
        index_extended
        and not middle_extended
        and not ring_extended
        and not little_extended
    ):
        return 1

    # --------------------------------------------------------
    # J2 - index + middle
    # --------------------------------------------------------
    if (
        index_extended
        and middle_extended
        and not ring_extended
        and not little_extended
    ):
        return 2

    # --------------------------------------------------------
    # J3 - index + middle + ring
    # --------------------------------------------------------
    if (
        index_extended
        and middle_extended
        and ring_extended
        and not little_extended
    ):
        return 3

    # --------------------------------------------------------
    # J4/J5 - all four main fingers extended
    # --------------------------------------------------------
    if (
        index_extended
        and middle_extended
        and ring_extended
        and little_extended
    ):
        if thumb_extended:
            return 5

        return 4

    return None

def is_joint_pointer_gesture(landmarks):
    """
    Joint selection gesture:

        LEFT index finger extended
        middle/ring/little folded

    Thumb is ignored.

    This gesture is used only in JOINT mode.
    """

    palm_center = calculate_palm_center(
        landmarks
    )

    index_extended = is_finger_extended(
        landmarks,
        8,
        6,
        palm_center,
    )

    middle_extended = is_finger_extended(
        landmarks,
        12,
        10,
        palm_center,
    )

    ring_extended = is_finger_extended(
        landmarks,
        16,
        14,
        palm_center,
    )

    little_extended = is_finger_extended(
        landmarks,
        20,
        18,
        palm_center,
    )

    return (
        index_extended
        and not middle_extended
        and not ring_extended
        and not little_extended
    )


def detect_joint_from_pointer(landmarks):
    """
    Use the LEFT index fingertip position to determine
    which on-screen J1-J6 box is being pointed at.

    Returns:
        1..6
        or None
    """

    if not is_joint_pointer_gesture(
        landmarks
    ):
        return None

    index_tip = landmarks[8]

    # Pointer must actually be inside selector area
    if not (
        JOINT_SELECTOR_X_MIN
        <= index_tip.x
        <= JOINT_SELECTOR_X_MAX
    ):
        return None

    if not (
        JOINT_SELECTOR_Y_MIN
        <= index_tip.y
        <= JOINT_SELECTOR_Y_MAX
    ):
        return None

    relative_x = (
        index_tip.x
        - JOINT_SELECTOR_X_MIN
    ) / (
        JOINT_SELECTOR_X_MAX
        - JOINT_SELECTOR_X_MIN
    )

    joint_index = int(
        relative_x * 6
    )

    joint_index = max(
        0,
        min(5, joint_index),
    )

    return joint_index + 1


def draw_joint_selector(
    frame,
    hovered_joint,
    selected_joint,
):
    """
    Draw J1-J6 selection boxes on the webcam.
    """

    height, width = frame.shape[:2]

    x_start = int(
        JOINT_SELECTOR_X_MIN * width
    )

    x_end = int(
        JOINT_SELECTOR_X_MAX * width
    )

    y_start = int(
        JOINT_SELECTOR_Y_MIN * height
    )

    y_end = int(
        JOINT_SELECTOR_Y_MAX * height
    )

    total_width = x_end - x_start
    box_width = total_width / 6.0

    for joint_number in range(1, 7):

        box_left = int(
            x_start
            + (joint_number - 1) * box_width
        )

        box_right = int(
            x_start
            + joint_number * box_width
        )

        # Default
        color = (255, 255, 255)
        thickness = 2

        # Currently selected joint
        if joint_number == selected_joint:
            color = (0, 255, 0)
            thickness = 4

        # Joint currently being pointed at
        if joint_number == hovered_joint:
            color = (0, 165, 255)
            thickness = 4

        cv2.rectangle(
            frame,
            (box_left, y_start),
            (box_right, y_end),
            color,
            thickness,
        )

        label = f"J{joint_number}"

        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            2,
        )[0]

        text_x = int(
            (box_left + box_right) / 2
            - text_size[0] / 2
        )

        text_y = int(
            (y_start + y_end) / 2
            + text_size[1] / 2
        )

        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )


def apply_axis_lock(robot_dx, robot_dy, robot_dz, axis_mode):
    """Keep only the selected RoboDK Cartesian displacement."""
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
    """Connect to RoboDK and retrieve the robot, target and base frame."""
    rdk = Robolink()

    robot = rdk.Item(ROBOT_NAME, ITEM_TYPE_ROBOT)
    target = rdk.Item(TARGET_NAME, ITEM_TYPE_TARGET)
    base_frame = rdk.Item(BASE_FRAME_NAME, ITEM_TYPE_FRAME)

    if not robot.Valid():
        raise RuntimeError(
            f'RoboDK robot "{ROBOT_NAME}" was not found.'
        )

    if not target.Valid():
        raise RuntimeError(
            f'RoboDK target "{TARGET_NAME}" was not found.'
        )

    if not base_frame.Valid():
        raise RuntimeError(
            f'RoboDK frame "{BASE_FRAME_NAME}" was not found.'
        )

    # Keep the target under the fixed robot-base frame.
    if not target.Parent().equals(base_frame):
        target.setParentStatic(base_frame)

    target.setAsCartesianTarget()

    # Use the base frame as the active robot reference.
    robot.setPoseFrame(base_frame)

    return rdk, robot, target, base_frame

def wrapped_joint_difference(new_value, current_value):
    """
    Calculate the shortest angular difference between two joint values.
    This avoids treating +179 and -179 degrees as a 358-degree jump.
    """
    return (new_value - current_value + 180.0) % 360.0 - 180.0


def get_joint_solutions_from_mat(solutions_mat):
    """
    RoboDK SolveIK_All usually returns a matrix:
    rows = robot joints
    columns = IK solutions

    This converts it into:
    [
        [j1, j2, j3, ...],
        [j1, j2, j3, ...],
        ...
    ]
    """
    solutions = []

    rows = solutions_mat.size(0)
    cols = solutions_mat.size(1)

    for col in range(cols):
        joint_solution = []

        for row in range(rows):
            joint_solution.append(float(solutions_mat[row, col]))

        solutions.append(joint_solution)

    return solutions


def joint_distance(solution_joints, current_joints):
    """
    Calculates how far one IK solution is from the current robot joints.
    Smaller value means smoother/closer robot posture.
    """
    total = 0.0

    for solution_joint, current_joint in zip(solution_joints, current_joints):
        diff = wrapped_joint_difference(solution_joint, current_joint)
        total += diff * diff

    return math.sqrt(total)


def limit_joint_step(current_joints, target_joints, max_step_deg):
    """
    Move only a small step from current joints toward target joints.
    This prevents sudden robot motion.
    """
    new_joints = []

    for current_joint, target_joint in zip(current_joints, target_joints):
        diff = wrapped_joint_difference(target_joint, current_joint)

        diff_limited = float(
            np.clip(
                diff,
                -max_step_deg,
                max_step_deg,
            )
        )

        new_joints.append(current_joint + diff_limited)

    return new_joints


def send_real_gripper_command(robot, gripper_state, opening):
    """
    REAL ROBOT GRIPPER EXECUTION:
    Sends gripper action when REAL_ROBOT_MODE and REAL_GRIPPER_COMMANDS_ENABLED are True.
    Calls RoboDK instructions "Gripper_Open" or "Gripper_Close".
    """
    if not REAL_ROBOT_MODE or not REAL_GRIPPER_COMMANDS_ENABLED:
        return

    instruction_name = "Gripper_Open" if gripper_state == "OPEN" else "Gripper_Close"
    try:
        robot.RunInstruction(instruction_name, INSTRUCTION_CALL_PROGRAM)
        print(f"REAL GRIPPER COMMAND EXECUTED: {instruction_name} (opening={opening:.2f})")
    except Exception as e:
        print(f"REAL GRIPPER COMMAND ERROR: {e}")


def move_robot_to_target(robot, target_pose, tool_pose):
    """
    Joint-continuous IK controller.

    Instead of using only one IK result, this:
    1. Reads the current robot joints.
    2. Gets all IK solutions.
    3. Selects the IK solution closest to the current joints.
    4. Moves only a small joint step toward that solution.
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

    valid_solutions = []

    for solution in all_solutions:
        if len(solution) >= robot_dof:
            valid_solutions.append(solution[:robot_dof])

    if not valid_solutions:
        return False, "IK FAILED - NO VALID SOLUTION"

    best_solution = min(
        valid_solutions,
        key=lambda solution: joint_distance(solution, current_joints)
    )

    max_joint_change = max(
        abs(wrapped_joint_difference(best, current))
        for best, current in zip(best_solution, current_joints)
    )

    joint_changes = [
        wrapped_joint_difference(best, current)
        for best, current
        in zip(best_solution, current_joints)
    ]

    if max_joint_change > MAX_JOINT_STEP_DEG:
        print("\n========== IK JUMP DEBUG ==========")

        print(
            "CURRENT:",
            [round(j, 2) for j in current_joints]
        )

        print(
            "IK BEST:",
            [round(j, 2) for j in best_solution]
        )

        print(
            "DIFF:",
            [round(j, 2) for j in joint_changes]
        )

        print(
            "MAX:",
            round(max_joint_change, 2),
            "deg"
        )

        print("===================================\n")
        if max_joint_change > MAX_JOINT_STEP_DEG:
            return (
                False,
                f"JOINT JUMP REJECTED: {max_joint_change:.1f} DEG",
            )

    
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

def set_gripper_opening(gripper, opening):
    """
    opening = 1.0 means fully open
    opening = 0.0 means fully closed
    """
    if gripper is None or not gripper.Valid():
        return

    opening = float(np.clip(opening, 0.0, 1.0))

    # If gripper is an articulated robot/mechanism, set joints
    if gripper.Type() == ITEM_TYPE_ROBOT:
        joint_value = (
            GRIPPER_CLOSED_VALUE
            + opening * (GRIPPER_OPEN_VALUE - GRIPPER_CLOSED_VALUE)
        )
        gripper.setJoints([joint_value])


def check_gripper_object_contact(
    rdk,
    gripper,
    grip_attach_frame,
    pick_object,
):
    """
    Check collision or proximity between gripper and object.

    Returns:
        any_contact, distance_mm
    """
    if pick_object is None or not pick_object.Valid():
        return False, 9999.0

    collision_detected = False
    if gripper is not None and gripper.Valid():
        try:
            collision_detected = (rdk.Collision(gripper, pick_object) > 0)
        except Exception:
            collision_detected = False

    distance_mm = 9999.0
    if grip_attach_frame is not None and grip_attach_frame.Valid():
        grip_pos = np.array(grip_attach_frame.PoseAbs().Pos(), dtype=np.float64)
        obj_pos = np.array(pick_object.PoseAbs().Pos(), dtype=np.float64)
        distance_mm = float(np.linalg.norm(obj_pos - grip_pos))

    near_detected = (distance_mm <= ATTACH_DISTANCE_MM)
    any_contact = collision_detected or near_detected

    return any_contact, distance_mm


def update_gripper_animation(
    rdk,
    gripper,
    grip_attach_frame,
    pick_object,
    current_opening,
    target_opening,
    contact_latched,
):
    """
    Smoothly animate opening toward target_opening and check contact.
    """
    if gripper is None or not gripper.Valid():
        return (
            target_opening,
            contact_latched,
            False,
        )

    difference = target_opening - current_opening

    # --------------------------------------------------------
    # Opening
    # --------------------------------------------------------
    if difference > 0.0:
        contact_latched = False

        if abs(difference) < GRIPPER_STEP:
            new_opening = target_opening
        else:
            new_opening = current_opening + GRIPPER_STEP

        new_opening = float(np.clip(new_opening, 0.0, 1.0))
        set_gripper_opening(gripper, new_opening)

        return (
            new_opening,
            contact_latched,
            False,
        )

    # --------------------------------------------------------
    # Contact already detected: hold current opening
    # --------------------------------------------------------
    if contact_latched:
        return (
            current_opening,
            True,
            True,
        )

    # --------------------------------------------------------
    # Already reached fully closed without touching anything
    # --------------------------------------------------------
    if current_opening <= target_opening:
        set_gripper_opening(gripper, target_opening)

        return (
            target_opening,
            False,
            False,
        )

    # --------------------------------------------------------
    # Adaptive closing: test one small closing step
    # --------------------------------------------------------
    previous_safe_opening = current_opening

    test_opening = max(
        target_opening,
        current_opening - CONTACT_CLOSE_STEP,
    )

    set_gripper_opening(gripper, test_opening)

    if CONTACT_SETTLE_SEC > 0:
        time.sleep(CONTACT_SETTLE_SEC)

    contact_detected, distance_mm = check_gripper_object_contact(
        rdk,
        gripper,
        grip_attach_frame,
        pick_object,
    )

    if contact_detected:
        set_gripper_opening(gripper, previous_safe_opening)
        print(
            f"Contact detected at distance={distance_mm:.1f} mm. "
            f"Holding opening={previous_safe_opening:.3f}"
        )
        return (
            previous_safe_opening,
            True,
            True,
        )

    return (
        test_opening,
        False,
        False,
    )

def smooth_and_limit_pose(previous_pose, desired_pose):
    """
    Smooth the requested TCP pose and limit the maximum TCP movement
    per update. Orientation stays the same; only position is smoothed.
    """
    previous_position = np.array(
        previous_pose.Pos(),
        dtype=np.float64,
    )

    desired_position = np.array(
        desired_pose.Pos(),
        dtype=np.float64,
    )

    # Exponential smoothing
    smoothed_position = (
        previous_position
        + TARGET_SMOOTHING_ALPHA
        * (desired_position - previous_position)
    )

    movement_vector = smoothed_position - previous_position
    movement_distance = np.linalg.norm(movement_vector)

    # Ignore very tiny movement
    if movement_distance < MIN_TCP_STEP_MM:
        return deepcopy(previous_pose)

    # Limit maximum TCP step
    if movement_distance > MAX_TCP_STEP_MM:
        movement_vector = (
            movement_vector / movement_distance
        ) * MAX_TCP_STEP_MM

        smoothed_position = previous_position + movement_vector

    new_pose = deepcopy(previous_pose)
    new_pose.setPos(smoothed_position.tolist())

    return new_pose

def enforce_safe_workspace(desired_pose, workspace_center):
    """
    Keep the TCP target inside a safe box around the starting TCP.
    Returns:
        safe_pose
        was_limited
    """
    desired_position = np.array(
        desired_pose.Pos(),
        dtype=np.float64,
    )

    center = np.array(
        workspace_center,
        dtype=np.float64,
    )

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

    inside_workspace = np.all(
        desired_position >= min_position
    ) and np.all(
        desired_position <= max_position
    )

    if inside_workspace:
        return desired_pose, False

    if not CLAMP_TO_WORKSPACE:
        return None, True

    limited_position = np.clip(
        desired_position,
        min_position,
        max_position,
    )

    safe_pose = deepcopy(desired_pose)
    safe_pose.setPos(limited_position.tolist())

    return safe_pose, True

def is_finger_extended(landmarks, tip_index, pip_index, palm_center):
    """
    Checks if a finger is extended.
    Fingertip should be farther from palm center than the middle finger joint.
    """
    tip_distance = math.hypot(
        landmarks[tip_index].x - palm_center[0],
        landmarks[tip_index].y - palm_center[1],
    )

    pip_distance = math.hypot(
        landmarks[pip_index].x - palm_center[0],
        landmarks[pip_index].y - palm_center[1],
    )

    return tip_distance > pip_distance * 1.25

def detect_gripper_toggle_gesture(landmarks):
    """
    RIGHT closed fist -> gripper toggle command.

    Open palm does nothing.

    A real pinch will normally have the other fingers extended,
    while a fist has all four main fingers folded.
    """

    palm_center = calculate_palm_center(
        landmarks
    )

    index_extended = is_finger_extended(
        landmarks, 8, 6, palm_center
    )

    middle_extended = is_finger_extended(
        landmarks, 12, 10, palm_center
    )

    ring_extended = is_finger_extended(
        landmarks, 16, 14, palm_center
    )

    little_extended = is_finger_extended(
        landmarks, 20, 18, palm_center
    )

    extended_count = sum([
        index_extended,
        middle_extended,
        ring_extended,
        little_extended,
    ])

    # Proper closed fist only
    return extended_count == 0

def item_position_abs(item):
    """
    Get absolute/global item position.
    """
    return np.array(
        item.PoseAbs().Pos(),
        dtype=np.float64,
    )


def update_gripper_pick_drop(
    grip_attach_frame,
    base_frame,
    pick_object,
    gripper_state,
    gripper_current_opening,
    gripper_contact_detected,
    held_object,
):
    """
    Simulated pick/drop for uFactory xArm Gripper.
    """

    if grip_attach_frame is None or not grip_attach_frame.Valid():
        return held_object

    if pick_object is None or not pick_object.Valid():
        return held_object

    # --------------------------------------------------------
    # Release object
    # --------------------------------------------------------
    if gripper_state == "OPEN" and held_object is not None:
        held_object.setParentStatic(base_frame)
        print(f"Released object: {held_object.Name()}")
        return None

    # --------------------------------------------------------
    # Grab object
    # --------------------------------------------------------
    if gripper_state == "CLOSE" and held_object is None:
        grip_position = np.array(
            grip_attach_frame.PoseAbs().Pos(),
            dtype=np.float64,
        )

        object_position = np.array(
            pick_object.PoseAbs().Pos(),
            dtype=np.float64,
        )

        distance_mm = float(
            np.linalg.norm(object_position - grip_position)
        )

        if distance_mm <= ATTACH_DISTANCE_MM:
            # Make object a child of gripping frame
            pick_object.setParentStatic(grip_attach_frame)

            # Snap to defined local holding offset
            pick_object.setPose(
                transl(
                    CUBE_HOLD_OFFSET_MM[0],
                    CUBE_HOLD_OFFSET_MM[1],
                    CUBE_HOLD_OFFSET_MM[2],
                )
            )

            print(
                f"Grabbed object: {pick_object.Name()} "
                f"at distance {distance_mm:.1f} mm"
            )

            return pick_object

    return held_object


def pose_distance_mm(pose_a, pose_b):
    """
    Distance between two TCP poses using only XYZ position.
    """
    pos_a = np.array(pose_a.Pos(), dtype=np.float64)
    pos_b = np.array(pose_b.Pos(), dtype=np.float64)

    return float(np.linalg.norm(pos_a - pos_b))


def add_record_sample(
    recorded_path,
    recording_start_time,
    robot,
    tcp_pose,
    gripper_state,
    held_object,
):
    """
    Save one robot path sample.
    """
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
    """
    Save recorded robot joints and TCP positions to CSV.
    """
    RECORDINGS_FOLDER.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RECORDINGS_FOLDER / f"xarm850_hand_recording_{timestamp}.csv"

    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "time_sec",
            "j1", "j2", "j3", "j4", "j5", "j6",
            "tcp_x", "tcp_y", "tcp_z",
            "gripper_state",
            "holding",
        ])

        for sample in recorded_path:
            joints = sample["joints"]
            position = sample["pose"].Pos()

            writer.writerow([
                round(sample["time"], 4),
                *[round(j, 4) for j in joints],
                round(position[0], 4),
                round(position[1], 4),
                round(position[2], 4),
                sample["gripper_state"],
                sample["holding"],
            ])

    print(f"Saved recording CSV: {csv_path}")

    return csv_path


def create_robodk_program_from_recording(
    rdk,
    robot,
    base_frame,
    recorded_path,
):
    """
    Create a RoboDK program from the recorded joint/TCP path.
    """
    if not recorded_path:
        print("No recorded path to create program.")
        return None

    timestamp = datetime.now().strftime("%H%M%S")
    program_name = f"{RECORDED_PROGRAM_NAME}_{timestamp}"

    program = rdk.AddProgram(program_name, robot)

    print(f'Creating RoboDK program: "{program_name}"')

    for index, sample in enumerate(recorded_path):
        target_name = f"{program_name}_T{index:03d}"

        target = rdk.AddTarget(
            target_name,
            base_frame,
            robot,
        )

        target.setPose(sample["pose"])
        target.setJoints(sample["joints"])

        program.MoveJ(target)

    print(
        f'RoboDK program "{program_name}" created with '
        f"{len(recorded_path)} points."
    )

    return program

def save_manual_waypoint(
    rdk,
    robot,
    base_frame,
    taught_waypoints,
    tcp_pose,
    gripper_state,
):
    """
    Save the current robot TCP pose as a manual taught waypoint.
    A RoboDK target is also created immediately in the station tree.
    """
    waypoint_index = len(taught_waypoints) + 1

    target_name = f"{TEACH_TARGET_PREFIX}_{waypoint_index:03d}"

    target = rdk.AddTarget(
        target_name,
        base_frame,
        robot,
    )

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

    print(
        f"Saved waypoint {waypoint_index}: "
        f"{target_name} | Gripper: {gripper_state}"
    )

    return taught_waypoints

def add_gripper_action_to_program(
    program,
    gripper_state,
):
    """
    Add the corresponding gripper action
    to a generated RoboDK program.
    """

    if gripper_state == "OPEN":

        program.RunInstruction(
            "Gripper_Open",
            INSTRUCTION_CALL_PROGRAM,
        )

        return True

    elif gripper_state == "CLOSE":

        program.RunInstruction(
            "Gripper_Close",
            INSTRUCTION_CALL_PROGRAM,
        )

        return True

    print(
        f"Warning: unknown gripper state: "
        f"{gripper_state}"
    )

    return False

def create_program_from_manual_waypoints(
    rdk,
    robot,
    taught_waypoints,
):
    """
    Create a RoboDK program from taught waypoints.

    The generated program contains:
        - Robot MoveJ instructions
        - Gripper Open/Close program calls

    Gripper commands are only added when the
    saved gripper state changes.
    """

    if not taught_waypoints:
        print("No taught waypoints available.")
        return None

    timestamp = datetime.now().strftime("%H%M%S")

    program_name = (
        f"{TEACH_PROGRAM_NAME}_{timestamp}"
    )

    program = rdk.AddProgram(
        program_name,
        robot,
    )

    print(
        f'Creating manual taught program: '
        f'"{program_name}"'
    )

    # --------------------------------------------------------
    # Initial gripper state
    # --------------------------------------------------------

    previous_gripper_state = (
        taught_waypoints[0]["gripper_state"]
    )

    add_gripper_action_to_program(
        program,
        previous_gripper_state,
    )

    print(
        f"Initial gripper state: "
        f"{previous_gripper_state}"
    )

    # --------------------------------------------------------
    # Add taught targets
    # --------------------------------------------------------

    for index, waypoint in enumerate(
        taught_waypoints,
        start=1,
    ):

        target = waypoint["target"]

        current_gripper_state = (
            waypoint["gripper_state"]
        )

        # --------------------------------------------
        # Move robot to taught target
        # --------------------------------------------

        program.MoveJ(target)

        print(
            f"Added waypoint {index}: "
            f"{target.Name()}"
        )

        # --------------------------------------------
        # Gripper state changed at this target
        # --------------------------------------------

        if (
            current_gripper_state
            != previous_gripper_state
        ):

            add_gripper_action_to_program(
                program,
                current_gripper_state,
            )

            print(
                f"Added gripper action after "
                f"{target.Name()}: "
                f"{current_gripper_state}"
            )

            previous_gripper_state = (
                current_gripper_state
            )

    print(
        f'Program "{program_name}" created with '
        f'{len(taught_waypoints)} taught targets '
        f'and automatic gripper actions.'
    )

    return program

def detect_save_waypoint_gesture(landmarks, pinch_active):
    """
    Detect V-sign gesture for saving a waypoint.

    Save gesture:
        index finger extended
        middle finger extended
        ring finger folded
        little finger folded

    Thumb is ignored.

    Important:
        If thumb-index pinch is active, saving is disabled
        so it does not conflict with robot movement.
    """
    if pinch_active:
        return False

    palm_center = calculate_palm_center(landmarks)

    index_extended = is_finger_extended(
        landmarks,
        8,   # index fingertip
        6,   # index PIP joint
        palm_center,
    )

    middle_extended = is_finger_extended(
        landmarks,
        12,  # middle fingertip
        10,  # middle PIP joint
        palm_center,
    )

    ring_extended = is_finger_extended(
        landmarks,
        16,  # ring fingertip
        14,  # ring PIP joint
        palm_center,
    )

    little_extended = is_finger_extended(
        landmarks,
        20,  # little fingertip
        18,  # little PIP joint
        palm_center,
    )

    palm_width = distance_2d(
        landmarks[5],
        landmarks[17],
    )

    index_middle_distance = distance_2d(
        landmarks[8],
        landmarks[12],
    )

    fingers_separated = (
        index_middle_distance
        > palm_width * V_SIGN_SEPARATION_RATIO
    )

    return (
        index_extended
        and middle_extended
        and not ring_extended
        and not little_extended
        and fingers_separated
    )

def is_new_waypoint_far_enough(taught_waypoints, tcp_pose):
    """
    Prevent saving duplicate waypoints at nearly the same TCP position.
    """
    if not taught_waypoints:
        return True

    last_pose = taught_waypoints[-1]["pose"]

    distance_mm = pose_distance_mm(
        last_pose,
        tcp_pose,
    )

    return distance_mm >= SAVE_TARGET_MIN_DISTANCE_MM


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model was not found: {MODEL_PATH.resolve()}"
        )

    # --------------------------------------------------------
    # Connect to RoboDK
    # --------------------------------------------------------

    rdk, robot, target, base_frame = connect_to_robodk()
    gripper = load_gripper(rdk)

    if gripper is not None and gripper.Valid():
        robot.setTool(gripper)

    pick_object = rdk.Item(PICK_OBJECT_NAME)

    if not pick_object.Valid():
        print(f'Warning: pick object "{PICK_OBJECT_NAME}" was not found.')
        pick_object = None
    else:
        print(f'Loaded pick object: "{pick_object.Name()}"')

    # --------------------------------------------------------
    # Load gripping attachment frame
    # --------------------------------------------------------

    grip_attach_frame = rdk.Item(
        GRIP_ATTACH_FRAME_NAME,
        ITEM_TYPE_FRAME,
    )

    if not grip_attach_frame.Valid():
        raise RuntimeError(
            f'Grip attach frame "{GRIP_ATTACH_FRAME_NAME}" was not found.'
        )

    if gripper is not None:
        print("uFactory xArm Gripper gesture control is active.")
    else:
        print("Gripper not found. Gesture will be detected but gripper will not move.")

    # Store the currently selected gripper TCP.
    tool_pose = robot.PoseTool()

    # Read the current uFactory 850 joints.
    starting_robot_joints = robot.Joints().list()

    # Read robot joint limits
    lower_lim, upper_lim, *_ = robot.JointLimits()
    joint_lower_limits = lower_lim.list() if hasattr(lower_lim, "list") else list(lower_lim)
    joint_upper_limits = upper_lim.list() if hasattr(upper_lim, "list") else list(upper_lim)

    # Calculate the current TCP pose relative to the robot base.
    starting_tcp_pose = robot.SolveFK(
        starting_robot_joints,
        tool=tool_pose,
        reference=eye(4),
    )

    workspace_center = starting_tcp_pose.Pos()

    print(
        "Safe workspace center:",
        [round(value, 2) for value in workspace_center],
    )

    # Place Hand_Target exactly on the current TCP.
    # This prevents the robot from jumping when the script starts.
    target.setPose(starting_tcp_pose)

    original_target_pose = starting_tcp_pose
    last_safe_target_pose = starting_tcp_pose
    last_robot_update_time = 0.0

    if REAL_ROBOT_MODE:
        robot.setSpeed(REAL_ROBOT_SPEED_MM_S)
        robot.setSpeedJoints(REAL_ROBOT_SPEED_JOINT_DEG_S)
        robot.setAcceleration(REAL_ROBOT_ACCEL_MM_S2)
        robot.setAccelerationJoints(REAL_ROBOT_ACCEL_JOINT_DEG_S2)
        print("REAL ROBOT PARAMETERS CONFIGURED:")
        print(f"  Linear Speed: {REAL_ROBOT_SPEED_MM_S} mm/s | Accel: {REAL_ROBOT_ACCEL_MM_S2} mm/s²")
        print(f"  Joint Speed: {REAL_ROBOT_SPEED_JOINT_DEG_S} deg/s | Accel: {REAL_ROBOT_ACCEL_JOINT_DEG_S2} deg/s²")
    print(f'Real Robot Mode: {"ENABLED (LIVE MOTION)" if REAL_ROBOT_MODE else "SIMULATION ONLY (Disabled)"}')
    print("Connected to RoboDK.")
    print(f'Controlling target: "{TARGET_NAME}"')
    print("The simulated uFactory 850 will follow Hand_Target.")
    print("The current TCP orientation will remain fixed.")
    print("Keep RoboDK disconnected from the physical robot.")

    # --------------------------------------------------------
    # Open webcam
    # --------------------------------------------------------

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {CAMERA_INDEX}."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # --------------------------------------------------------
    # Configure MediaPipe
    # --------------------------------------------------------

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.60,
        min_hand_presence_confidence=0.60,
        min_tracking_confidence=0.60,
    )

    filtered_palm = None

    # ============================================================
    # ROBUST PINCH STATE
    # ============================================================

    # Stable confirmed pinch state
    pinch_state = False

    # Consecutive-frame counters
    pinch_engage_count = 0
    pinch_release_count = 0

    # Existing movement state
    pinch_previous = False

    cartesian_clutch_active = False
    hand_anchor = None
    target_anchor_position = None
    hand_size_anchor = None
    filtered_hand_size = None

    # Safe gripper-toggle state
    gripper_toggle_start_time = None
    gripper_toggle_latched = False
    gripper_release_count = 0
    gripper_toggle_status = "READY"

    gripper_state = "OPEN"
    gripper_target_opening = 1.0
    gripper_current_opening = 1.0
    gripper_contact_latched = False
    left_contact = False
    right_contact = False

    # Left-hand axis-selection state
    axis_mode = AXIS_MODE_DEFAULT
    axis_candidate = None
    axis_candidate_count = 0
    axis_hand_present = False
# Control-mode state
    control_mode = CONTROL_MODE_DEFAULT
    mode_gesture_start_time = None
    mode_gesture_candidate = None
    mode_gesture_status = "READY"
    
    # After entering Cartesian mode, require right pinch to be
    # released once before Cartesian movement is enabled.
    cartesian_rearm_required = False

    # Joint-mode state
    selected_joint = None

    # Direct on-screen selector
    joint_selector_candidate = None
    joint_selector_start_time = None
    joint_selector_latched = False
    joint_selector_release_count = 0
    joint_selector_status = "POINT TO J1-J6"

    # Prevent accidental movement when entering
    # Joint Mode or selecting a new joint
    joint_rearm_required = False

    # Right-hand joint movement
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

    print()
    print("Controls:")
    print("  RIGHT HAND pinch: move Hand_Target")
    print("  LEFT HAND fist: pause movement")
    print("  LEFT HAND 1 finger: X-axis only")
    print("  LEFT HAND 2 fingers: Y-axis only")
    print("  LEFT HAND 3 fingers: Z-axis only")
    print("  LEFT HAND 4 fingers: free XYZ movement")
    print("  Release right-hand pinch: freeze Hand_Target")
    print("  R: reset target")
    print("  Q or Esc: exit")
    print("  S: start/stop path recording")
    print("  T: save current robot position as target")
    print("  G: generate RoboDK program from taught targets")
    print("  X: clear taught targets")
    
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                success, frame = camera.read()

                if not success:
                    print("Warning: webcam frame could not be read.")
                    continue

                # Mirror the view so movement feels natural.
                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                media_pipe_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int(
                    (time.perf_counter() - start_time) * 1000
                )

                # MediaPipe requires strictly increasing timestamps.
                if timestamp_ms <= previous_timestamp_ms:
                    timestamp_ms = previous_timestamp_ms + 1

                previous_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(
                    media_pipe_image,
                    timestamp_ms,
                )

                tracking_status = "NO RIGHT HAND"
                status_color = (0, 0, 255)

                # Per-frame defaults
                pinch_active = False
                pinch_ratio = 0.0
                palm_pixel = None
                save_gesture_detected = False
                save_gesture_status = "OFF"
                axis_hand_present = False
                joint_hovered = None

                hands_by_label = get_hands_by_label(result)

                control_landmarks = hands_by_label.get(
                    CONTROL_HAND_LABEL
                )
                axis_landmarks = hands_by_label.get(
                    AXIS_HAND_LABEL
                )

                # Draw both hands with different colours.
                if control_landmarks is not None:
                    draw_hand(
                        frame,
                        control_landmarks,
                        label="RIGHT: CONTROL",
                        line_color=(255, 255, 255),
                    )

                if axis_landmarks is not None:
                    draw_hand(
                        frame,
                        axis_landmarks,
                        label="LEFT: AXIS",
                        line_color=(0, 165, 255),
                    )

                # ------------------------------------------------
                # Left hand:
                # 1. Select Cartesian / Joint mode
                # 2. In Cartesian mode, select movement axis
                # ------------------------------------------------
                axis_changed = False
                mode_gesture_status = "READY"

                if axis_landmarks is not None:
                    axis_hand_present = True

                    # ------------------------------------------------
                    # Check thumbs-up / thumbs-down first
                    # ------------------------------------------------

                    detected_control_mode = detect_control_mode_gesture(
                        axis_landmarks
                    )

                    if detected_control_mode is not None:

                        current_mode_time = time.perf_counter()

                        if detected_control_mode != mode_gesture_candidate:
                            mode_gesture_candidate = detected_control_mode
                            mode_gesture_start_time = current_mode_time

                        elif mode_gesture_start_time is not None:

                            mode_hold_time = (
                                current_mode_time - mode_gesture_start_time
                            )

                            mode_gesture_status = (
                                f"HOLD {detected_control_mode} "
                                f"{mode_hold_time:.1f}s"
                            )

                            if mode_hold_time >= MODE_SWITCH_HOLD_SEC:

                                if control_mode != detected_control_mode:
                                    control_mode = detected_control_mode

                                    print(
                                        f"Control mode changed to: "
                                        f"{control_mode}"
                                    )

                                    # Stop any active Cartesian movement.
                                    pinch_previous = False
                                    hand_anchor = None
                                    target_anchor_position = None
                                    hand_size_anchor = None

                                    axis_candidate = None
                                    axis_candidate_count = 0

                                    mode_gesture_status = (
                                        f"MODE: {control_mode}"
                                    )
                                    # ------------------------------------------------
                                    # Stop all previous movement/clutch states
                                    # ------------------------------------------------
                                    pinch_previous = False
                                    cartesian_clutch_active = False
                                    hand_anchor = None
                                    target_anchor_position = None
                                    hand_size_anchor = None

                                    joint_hand_anchor_x = None
                                    joint_angle_anchor = None

                                    axis_candidate = None
                                    axis_candidate_count = 0

                                    # ------------------------------------------------
                                    # IMPORTANT:
                                    # Synchronize Cartesian target with the robot's
                                    # ACTUAL current joint position.
                                    # ------------------------------------------------
                                    current_joints = robot.Joints().list()

                                    current_tcp_pose = robot.SolveFK(
                                        current_joints,
                                        tool=tool_pose,
                                        reference=eye(4),
                                    )

                                    target.setPose(current_tcp_pose)

                                    last_safe_target_pose = deepcopy(
                                        current_tcp_pose
                                    )

                                    # ============================================================
                                    # ENTERING CARTESIAN MODE
                                    # ============================================================

                                    if control_mode == "CARTESIAN":

                                        # --------------------------------------------------------
                                        # Re-center Cartesian workspace around CURRENT robot pose
                                        # --------------------------------------------------------

                                        workspace_center = (
                                            current_tcp_pose.Pos()
                                        )

                                        print(
                                            "Cartesian workspace re-centered at:",
                                            [
                                                round(value, 2)
                                                for value in workspace_center
                                            ],
                                        )

                                        # Start Cartesian mode paused
                                        axis_mode = "PAUSE"

                                        # Require right pinch release before movement
                                        cartesian_rearm_required = True

                                        print(
                                            "Cartesian mode armed at current robot pose. "
                                            "Release right pinch, then select an axis."
                                        )

                                        mode_gesture_status = (
                                            "MODE: CARTESIAN"
                                        )


                                    elif control_mode == "JOINT":

                                        # Start Joint Mode with no selected joint
                                        selected_joint = None

                                        # Reset selector ONCE when entering Joint Mode
                                        joint_selector_candidate = None
                                        joint_selector_start_time = None
                                        joint_selector_latched = False
                                        joint_selector_release_count = 0

                                        joint_selector_status = (
                                            "POINT TO J1-J6"
                                        )

                                        # Require right pinch release before movement
                                        joint_rearm_required = True

                                        joint_hand_anchor_x = None
                                        joint_angle_anchor = None

                                        pinch_previous = False

                                        print(
                                            "Joint Mode ready. "
                                            "Point to J1-J6 to select a joint."
                                        )

                                        mode_gesture_status = (
                                            "MODE: JOINT"
                                        )

                                    # ============================================================
                                    # Require mode gesture release before another mode command
                                    # ============================================================

                                    mode_gesture_start_time = None

                    else:
                        mode_gesture_candidate = None
                        mode_gesture_start_time = None

                    # ------------------------------------------------
                    # Cartesian mode: normal axis-selection gestures
                    # ------------------------------------------------

                    if control_mode == "CARTESIAN":

                        detected_axis_mode = detect_axis_mode(
                            axis_landmarks
                        )

                        if detected_axis_mode is None:
                            axis_candidate = None
                            axis_candidate_count = 0

                        elif detected_axis_mode == axis_candidate:
                            axis_candidate_count += 1

                        else:
                            axis_candidate = detected_axis_mode
                            axis_candidate_count = 1

                        if (
                            axis_candidate_count
                            >= AXIS_DEBOUNCE_FRAMES
                            and detected_axis_mode != axis_mode
                        ):
                            axis_mode = detected_axis_mode
                            axis_changed = True

                            # Reset movement anchor when axis changes.
                            pinch_previous = False
                            cartesian_clutch_active = False
                            hand_anchor = None
                            target_anchor_position = None
                            hand_size_anchor = None

                            print(
                                f"Cartesian axis mode changed to: "
                                f"{axis_mode}"
                            )

                    elif control_mode == "JOINT":

                        # ------------------------------------------------
                        # Direct J1-J6 pointer selection
                        # ------------------------------------------------

                        joint_hovered = detect_joint_from_pointer(
                            axis_landmarks
                        )

                        current_selector_time = (
                            time.perf_counter()
                        )

                        # ------------------------------------------------
                        # A joint was already selected.
                        # Require pointer release before another selection.
                        # ------------------------------------------------
                        if joint_selector_latched:

                            if joint_hovered is None:

                                joint_selector_release_count += 1

                                if (
                                    joint_selector_release_count
                                    >= JOINT_SELECTOR_RELEASE_FRAMES
                                ):
                                    joint_selector_latched = False
                                    joint_selector_release_count = 0

                                    joint_selector_status = (
                                        "POINT TO J1-J6"
                                    )

                            else:
                                joint_selector_release_count = 0

                                joint_selector_status = (
                                    f"J{selected_joint} SELECTED - "
                                    f"REMOVE POINTER"
                                )

                        # ------------------------------------------------
                        # Ready to select a joint
                        # ------------------------------------------------
                        else:

                            if joint_hovered is None:

                                joint_selector_candidate = None
                                joint_selector_start_time = None

                                joint_selector_status = (
                                    "POINT TO J1-J6"
                                )

                            else:

                                # New box
                                if (
                                    joint_hovered
                                    != joint_selector_candidate
                                ):

                                    joint_selector_candidate = (
                                        joint_hovered
                                    )

                                    joint_selector_start_time = (
                                        current_selector_time
                                    )

                                elif (
                                    joint_selector_start_time
                                    is not None
                                ):

                                    selector_hold_time = (
                                        current_selector_time
                                        - joint_selector_start_time
                                    )

                                    joint_selector_status = (
                                        f"HOLD J{joint_hovered}: "
                                        f"{selector_hold_time:.1f}/"
                                        f"{JOINT_SELECTOR_HOLD_SEC:.1f}s"
                                    )

                                    # ------------------------------------
                                    # Selection confirmed
                                    # ------------------------------------
                                    if (
                                        selector_hold_time
                                        >= JOINT_SELECTOR_HOLD_SEC
                                    ):

                                        selected_joint = joint_hovered

                                        print(
                                            f"Selected robot joint: "
                                            f"J{selected_joint}"
                                        )

                                        joint_selector_latched = True
                                        joint_selector_release_count = 0

                                        joint_selector_candidate = None
                                        joint_selector_start_time = None

                                        joint_selector_status = (
                                            f"J{selected_joint} SELECTED"
                                        )

                                        # Reset movement reference
                                        joint_hand_anchor_x = None
                                        joint_angle_anchor = None
                                        pinch_previous = False

                                        # IMPORTANT:
                                        # if right hand is already pinching,
                                        # don't immediately move the new joint.
                                        joint_rearm_required = True

                else:
                    axis_candidate = None
                    axis_candidate_count = 0

                    if REQUIRE_AXIS_HAND:
                        pinch_previous = False
                        hand_anchor = None
                        target_anchor_position = None
                        hand_size_anchor = None

                # ------------------------------------------------
                # Right hand: movement, gripper and waypoint save
                # ------------------------------------------------
                if control_landmarks is not None:
                    landmarks = control_landmarks

                    raw_palm = calculate_palm_center(landmarks)

                    if filtered_palm is None:
                        filtered_palm = raw_palm.copy()
                    else:
                        filtered_palm = (
                            SMOOTHING_ALPHA * raw_palm
                            + (1.0 - SMOOTHING_ALPHA)
                            * filtered_palm
                        )

                    palm_width = distance_2d(
                        landmarks[5],
                        landmarks[17],
                    )

                    if filtered_hand_size is None:
                        filtered_hand_size = palm_width
                    else:
                        filtered_hand_size = (
                            DEPTH_SMOOTHING_ALPHA * palm_width
                            + (1.0 - DEPTH_SMOOTHING_ALPHA)
                            * filtered_hand_size
                        )

                    thumb_index_distance = distance_2d(
                        landmarks[4],
                        landmarks[8],
                    )

                    index_length = distance_2d(
                        landmarks[5],
                        landmarks[8],
                    )

                    pinch_ratio = (
                        thumb_index_distance
                        / max(index_length, 0.0001)
                    )

                    # ============================================================
                    # ROBUST PINCH HYSTERESIS
                    # ============================================================

                    # ------------------------------------------------------------
                    # Currently NOT pinching
                    # ------------------------------------------------------------
                    if not pinch_state:

                        # Require a clearly closed pinch
                        if pinch_ratio <= PINCH_ENGAGE_THRESHOLD:

                            pinch_engage_count += 1

                            # Enough stable frames -> activate pinch
                            if pinch_engage_count >= PINCH_ENGAGE_FRAMES:

                                pinch_state = True

                                pinch_engage_count = 0
                                pinch_release_count = 0

                                print(
                                    f"PINCH ENGAGED | ratio={pinch_ratio:.2f}"
                                )

                        else:

                            # Any unstable frame cancels the pending engagement
                            pinch_engage_count = 0


                    # ------------------------------------------------------------
                    # Currently pinching
                    # ------------------------------------------------------------
                    else:

                        # Require the fingers to clearly separate before releasing
                        if pinch_ratio >= PINCH_RELEASE_THRESHOLD:

                            pinch_release_count += 1

                            # Enough stable release frames -> deactivate pinch
                            if pinch_release_count >= PINCH_RELEASE_FRAMES:

                                pinch_state = False

                                pinch_release_count = 0
                                pinch_engage_count = 0

                                print(
                                    f"PINCH RELEASED | ratio={pinch_ratio:.2f}"
                                )

                        else:

                            # Small noisy changes must NOT cancel the pinch
                            pinch_release_count = 0


                    # Confirmed pinch state used by the rest of the program
                    pinch_active = pinch_state

                    # ------------------------------------------------
                    # Waypoint V-sign detection has priority
                    # ------------------------------------------------

                    if ENABLE_GESTURE_WAYPOINT_SAVE:
                        save_gesture_detected = (
                            detect_save_waypoint_gesture(
                                landmarks,
                                pinch_active,
                            )
                        )

                    # ------------------------------------------------
                    # SAFE LATCHED GRIPPER CONTROL
                    # ------------------------------------------------

                    fist_for_gripper = False

                    if (
                        ENABLE_GRIPPER_GESTURES
                        and not save_gesture_detected
                    ):
                        fist_for_gripper = (
                            detect_gripper_toggle_gesture(
                                landmarks
                            )
                        )

                    current_gripper_time = time.perf_counter()

                    # ------------------------------------------------
                    # FIST DETECTED
                    # ------------------------------------------------
                    if fist_for_gripper:

                        # We are still seeing the fist,
                        # so release counter must stay at zero.
                        gripper_release_count = 0

                        # ------------------------------------------------
                        # Command already executed.
                        # Do not execute another one until fist is released.
                        # ------------------------------------------------
                        if gripper_toggle_latched:

                            gripper_toggle_status = (
                                "LOCKED - RELEASE FIST"
                            )

                        else:

                            # ------------------------------------------------
                            # Start hold timer only once
                            # ------------------------------------------------
                            if gripper_toggle_start_time is None:
                                gripper_toggle_start_time = (
                                    current_gripper_time
                                )

                            gripper_hold_time = (
                                current_gripper_time
                                - gripper_toggle_start_time
                            )

                            # ------------------------------------------------
                            # Decide which action is pending
                            # ------------------------------------------------
                            if gripper_state == "OPEN":

                                required_hold_time = (
                                    GRIPPER_CLOSE_HOLD_SEC
                                )

                                pending_action = "CLOSE"

                            else:

                                required_hold_time = (
                                    GRIPPER_OPEN_HOLD_SEC
                                )

                                pending_action = "OPEN"

                            gripper_toggle_status = (
                                f"HOLD FIST FOR {pending_action}: "
                                f"{gripper_hold_time:.1f}/"
                                f"{required_hold_time:.1f}s"
                            )

                            # ------------------------------------------------
                            # Hold time reached -> execute ONCE
                            # ------------------------------------------------
                            if (
                                gripper_hold_time
                                >= required_hold_time
                            ):

                                if gripper_state == "OPEN":

                                    gripper_state = "CLOSE"
                                    gripper_target_opening = 0.0

                                    print(
                                        "uFactory xArm Gripper command: CLOSE"
                                    )
                                    send_real_gripper_command(robot, gripper_state, gripper_target_opening)

                                else:

                                    gripper_state = "OPEN"
                                    gripper_target_opening = 1.0

                                    print(
                                        "uFactory xArm Gripper command: OPEN"
                                    )
                                    send_real_gripper_command(robot, gripper_state, gripper_target_opening)

                                # Lock command until fist is released
                                gripper_toggle_latched = True

                                gripper_toggle_start_time = None

                                gripper_toggle_status = (
                                    f"{gripper_state} COMMAND ACCEPTED"
                                )


                    # ------------------------------------------------
                    # FIST NOT DETECTED
                    # ------------------------------------------------
                    else:

                        # Cancel an unfinished fist hold
                        gripper_toggle_start_time = None

                        # ------------------------------------------------
                        # If a command was executed previously,
                        # wait for several non-fist frames before
                        # allowing another command.
                        # ------------------------------------------------
                        if gripper_toggle_latched:

                            gripper_release_count += 1

                            if (
                                gripper_release_count
                                >= GRIPPER_RELEASE_FRAMES
                            ):

                                gripper_toggle_latched = False
                                gripper_release_count = 0

                                gripper_toggle_status = "READY"

                        else:

                            gripper_release_count = 0
                            gripper_toggle_status = "READY"


                    height, width = frame.shape[:2]
                    palm_pixel = (
                        int(filtered_palm[0] * width),
                        int(filtered_palm[1] * height),
                    )

                    if control_mode == "CARTESIAN":
                        if (
                            cartesian_rearm_required
                            and not pinch_active
                        ):
                            cartesian_rearm_required = False

                            print(
                                "Cartesian control ready. "
                                "Select axis and pinch to move."
                            )

                        movement_allowed = (
                            not cartesian_rearm_required
                            and pinch_active
                            and not fist_for_gripper
                            and axis_mode != "PAUSE"
                            and (
                                axis_hand_present
                                or not REQUIRE_AXIS_HAND
                            )
                        )
                        

                        if pinch_active and not movement_allowed:

                            if cartesian_rearm_required:
                                tracking_status = (
                                    "RELEASE RIGHT PINCH TO ARM CARTESIAN"
                                )

                            elif axis_mode == "PAUSE":
                                tracking_status = (
                                    "PINCH ACTIVE - LEFT HAND PAUSED"
                                )

                            else:
                                tracking_status = (
                                    "PINCH ACTIVE - SHOW LEFT AXIS HAND"
                                )

                            status_color = (0, 165, 255)

                            pinch_previous = False
                            hand_anchor = None
                            target_anchor_position = None
                            hand_size_anchor = None
                            cartesian_clutch_active = False

                            status_color = (0, 165, 255)
                            pinch_previous = False
                            hand_anchor = None
                            target_anchor_position = None
                            hand_size_anchor = None

                        elif movement_allowed:
                            tracking_status = (
                                f"PINCH ACTIVE - {axis_mode} MODE"
                            )
                            status_color = (0, 255, 0)

                            # ------------------------------------------------
                            # Create Cartesian clutch anchor ONCE
                            # ------------------------------------------------

                            if not cartesian_clutch_active:
                                # Read actual robot pose
                                current_robot_joints = robot.Joints().list()

                                anchor_pose = robot.SolveFK(
                                    current_robot_joints,
                                    tool=tool_pose,
                                    reference=eye(4),
                                )

                                # Synchronize Hand_Target with robot
                                target.setPose(anchor_pose)

                                # Current robot pose becomes Cartesian reference
                                last_safe_target_pose = deepcopy(
                                    anchor_pose
                                )
                                # Save hand position only ONCE
                                hand_anchor = filtered_palm.copy()
                                hand_size_anchor = filtered_hand_size

                                # Save target position only ONCE
                                target_anchor_position = (
                                    anchor_pose.Pos()
                                )

                                # Important:
                                # do not create another anchor until
                                # Cartesian clutch is released/reset.
                                cartesian_clutch_active = True

                                print(
                                    "Cartesian clutch STARTED from actual robot pose:",
                                    [
                                        round(value, 2)
                                        for value in target_anchor_position
                                    ],
                                )

                            if (
                                hand_anchor is not None
                                and target_anchor_position is not None
                            ):
                                hand_delta = (
                                    filtered_palm - hand_anchor
                                )

                                horizontal_delta = hand_delta[0]
                                vertical_delta = hand_delta[1]

                                if abs(horizontal_delta) < DEAD_ZONE:
                                    horizontal_delta = 0.0
                                if abs(vertical_delta) < DEAD_ZONE:
                                    vertical_delta = 0.0

                                # Webcam right/left controls RoboDK Y.
                                robot_dy = (
                                    horizontal_delta * X_SCALE_MM
                                )

                                # Webcam up/down controls RoboDK Z.
                                if vertical_delta < 0:
                                    robot_dz = (
                                        -vertical_delta
                                        * Z_UP_SCALE_MM
                                    )
                                else:
                                    robot_dz = (
                                        -vertical_delta
                                        * Z_DOWN_SCALE_MM
                                    )

                                robot_dy = float(
                                    np.clip(
                                        robot_dy,
                                        -MAX_X_MOVEMENT_MM,
                                        MAX_X_MOVEMENT_MM,
                                    )
                                )
                                robot_dz = float(
                                    np.clip(
                                        robot_dz,
                                        -MAX_Z_MOVEMENT_MM,
                                        MAX_Z_MOVEMENT_MM,
                                    )
                                )

                                # Hand depth controls RoboDK X.
                                robot_dx = 0.0

                                if (
                                    ENABLE_WEBCAM_DEPTH
                                    and hand_size_anchor is not None
                                ):
                                    size_ratio = (
                                        filtered_hand_size
                                        / max(
                                            hand_size_anchor,
                                            0.0001,
                                        )
                                    )

                                    robot_dx = (
                                        (size_ratio - 1.0)
                                        * Y_SCALE_MM
                                        * Y_DIRECTION
                                    )
                                    robot_dx = float(
                                        np.clip(
                                            robot_dx,
                                            -MAX_Y_MOVEMENT_MM,
                                            MAX_Y_MOVEMENT_MM,
                                        )
                                    )

                                (
                                    robot_dx,
                                    robot_dy,
                                    robot_dz,
                                ) = apply_axis_lock(
                                    robot_dx,
                                    robot_dy,
                                    robot_dz,
                                    axis_mode,
                                )

                                command_position = [
                                    target_anchor_position[0]
                                    + robot_dx,
                                    target_anchor_position[1]
                                    + robot_dy,
                                    target_anchor_position[2]
                                    + robot_dz,
                                ]

                                command_pose = deepcopy(
                                    last_safe_target_pose
                                )

                                command_pose.setPos(
                                    command_position
                                )

                                current_time = time.perf_counter()

                                if (
                                    current_time
                                    - last_robot_update_time
                                    >= ROBOT_UPDATE_INTERVAL
                                ):
                                    (
                                        safe_command_pose,
                                        workspace_limited,
                                    ) = enforce_safe_workspace(
                                        command_pose,
                                        workspace_center,
                                    )

                                    if safe_command_pose is None:
                                        accepted = False
                                        robot_status = (
                                            "OUTSIDE SAFE WORKSPACE"
                                        )
                                        smooth_pose = (
                                            last_safe_target_pose
                                        )
                                    else:
                                        smooth_pose = (
                                            smooth_and_limit_pose(
                                                last_safe_target_pose,
                                                safe_command_pose,
                                            )
                                        )

                                        (
                                            accepted,
                                            robot_status,
                                        ) = move_robot_to_target(
                                            robot,
                                            smooth_pose,
                                            tool_pose,
                                        )

                                        if (
                                            workspace_limited
                                            and accepted
                                        ):
                                            robot_status = (
                                                "WORKSPACE LIMIT ACTIVE"
                                            )

                                    last_robot_update_time = current_time

                                    if accepted:
                                        target.setPose(smooth_pose)
                                        last_safe_target_pose = (
                                            smooth_pose
                                        )
                                        status_color = (0, 255, 0)
                                    else:
                                        target.setPose(
                                            last_safe_target_pose
                                        )
                                        status_color = (0, 0, 255)
                                        tracking_status = robot_status

                                cv2.putText(
                                    frame,
                                    (
                                        f"Applied {axis_mode}: "
                                        f"X={robot_dx:+.1f} mm  "
                                        f"Y={robot_dy:+.1f} mm  "
                                        f"Z={robot_dz:+.1f} mm"
                                    ),
                                    (20, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 0),
                                    2,
                                )

                            pinch_previous = True

                        else:
                            tracking_status = (
                                "RIGHT HAND FOUND - PINCH TO MOVE"
                            )
                            status_color = (0, 255, 255)
                            pinch_previous = False
                            cartesian_clutch_active = False
                            hand_anchor = None
                            target_anchor_position = None
                            hand_size_anchor = None

                    # ========================================================
                    # JOINT MODE - RIGHT HAND PINCH CONTROL
                    # ========================================================

                    elif control_mode == "JOINT":

                        # ----------------------------------------------------
                        # SAFETY RE-ARM
                        #
                        # When:
                        #   - entering Joint Mode
                        #   - selecting a new joint
                        #
                        # the right-hand pinch must first be released.
                        # This prevents accidental joint movement.
                        # ----------------------------------------------------

                        if joint_rearm_required:

                            if not pinch_active:

                                # Right pinch has been released.
                                # Joint movement can now be armed.
                                joint_rearm_required = False

                                joint_hand_anchor_x = None
                                joint_angle_anchor = None

                                pinch_previous = False

                                print(
                                    "Joint movement armed. "
                                    "New right-hand pinch required."
                                )

                            else:

                                # Right hand was already pinching when
                                # Joint Mode / joint selection happened.
                                # Do NOT move the robot.
                                tracking_status = (
                                    "JOINT MODE - RELEASE RIGHT PINCH"
                                )

                                status_color = (0, 165, 255)

                                joint_hand_anchor_x = None
                                joint_angle_anchor = None

                                pinch_previous = False


                        # ====================================================
                        # MOVE SELECTED JOINT
                        # ====================================================

                        if (
                            not joint_rearm_required
                            and pinch_active
                            and not fist_for_gripper
                            and selected_joint is not None
                        ):

                            tracking_status = (
                                f"JOINT MODE - MOVING J{selected_joint}"
                            )

                            status_color = (0, 165, 255)

                            # ------------------------------------------------
                            # New right-hand pinch
                            # ------------------------------------------------

                            if (
                                not pinch_previous
                                or joint_hand_anchor_x is None
                                or joint_angle_anchor is None
                            ):

                                # Save current hand position
                                joint_hand_anchor_x = (
                                    filtered_palm[0]
                                )

                                # Read actual current robot joints
                                current_joints = (
                                    robot.Joints().list()
                                )

                                # Save current angle of selected joint
                                joint_angle_anchor = (
                                    current_joints[
                                        selected_joint - 1
                                    ]
                                )

                                print(
                                    f"J{selected_joint} movement started at "
                                    f"{joint_angle_anchor:.2f} deg"
                                )

                            # ------------------------------------------------
                            # Calculate horizontal hand displacement
                            # ------------------------------------------------

                            horizontal_delta = (
                                filtered_palm[0]
                                - joint_hand_anchor_x
                            )

                            if (
                                abs(horizontal_delta)
                                < DEAD_ZONE
                            ):
                                horizontal_delta = 0.0

                            # Convert hand movement into degrees
                            requested_joint_change = (
                                horizontal_delta
                                * JOINT_HAND_SCALE_DEG
                            )

                            requested_joint_change = float(
                                np.clip(
                                    requested_joint_change,
                                    -JOINT_MAX_FROM_ANCHOR_DEG,
                                    JOINT_MAX_FROM_ANCHOR_DEG,
                                )
                            )

                            desired_joint_angle = (
                                joint_angle_anchor
                                + requested_joint_change
                            )

                            joint_index = (
                                selected_joint - 1
                            )

                            # ------------------------------------------------
                            # Respect RoboDK joint limits
                            # ------------------------------------------------

                            desired_joint_angle = float(
                                np.clip(
                                    desired_joint_angle,
                                    joint_lower_limits[
                                        joint_index
                                    ],
                                    joint_upper_limits[
                                        joint_index
                                    ],
                                )
                            )

                            current_time = (
                                time.perf_counter()
                            )

                            # ------------------------------------------------
                            # Limit update frequency
                            # ------------------------------------------------

                            if (
                                current_time
                                - last_joint_update_time
                                >= JOINT_UPDATE_INTERVAL
                            ):

                                current_joints = (
                                    robot.Joints().list()
                                )

                                current_angle = (
                                    current_joints[
                                        joint_index
                                    ]
                                )

                                angle_difference = (
                                    desired_joint_angle
                                    - current_angle
                                )

                                angle_step = float(
                                    np.clip(
                                        angle_difference,
                                        -JOINT_CONTROL_STEP_DEG,
                                        JOINT_CONTROL_STEP_DEG,
                                    )
                                )

                                # ------------------------------------------------
                                # Change ONLY selected joint
                                # ------------------------------------------------

                                current_joints[
                                    joint_index
                                ] = (
                                    current_angle
                                    + angle_step
                                )

                                if REAL_ROBOT_MODE:
                                    robot.MoveJ(current_joints)
                                else:
                                    robot.setJoints(current_joints)

                                # ------------------------------------------------
                                # Synchronize Cartesian target with
                                # the robot's new actual TCP pose
                                # ------------------------------------------------

                                current_tcp_pose = robot.SolveFK(
                                    current_joints,
                                    tool=tool_pose,
                                    reference=eye(4),
                                )

                                target.setPose(
                                    current_tcp_pose
                                )

                                last_safe_target_pose = (
                                    deepcopy(
                                        current_tcp_pose
                                    )
                                )

                                last_joint_update_time = (
                                    current_time
                                )

                                print(
                                    f"J{selected_joint}: "
                                    f"{current_joints[joint_index]:.2f} deg"
                                )

                            pinch_previous = True


                        # ====================================================
                        # PINCH ACTIVE BUT NO JOINT SELECTED
                        # ====================================================

                        elif (
                            not joint_rearm_required
                            and pinch_active
                            and selected_joint is None
                        ):

                            tracking_status = (
                                "JOINT MODE - SELECT J1-J6 FIRST"
                            )

                            status_color = (
                                0,
                                165,
                                255,
                            )

                            joint_hand_anchor_x = None
                            joint_angle_anchor = None

                            pinch_previous = False

                    elif (
                        pinch_active
                        and selected_joint is None
                    ):
                        tracking_status = "JOINT MODE - SELECT A JOINT (J1-J6)"
                        status_color = (0, 165, 255)
                        pinch_previous = False

                    if (
                        control_mode == "JOINT"
                        and not pinch_active
                    ):
                        joint_hand_anchor_x = None
                        joint_angle_anchor = None
                        if selected_joint is not None:
                            tracking_status = f"JOINT MODE - J{selected_joint} SELECTED"
                        else:
                            tracking_status = "JOINT MODE - SELECT A JOINT"
                        status_color = (0, 255, 255)
                        pinch_previous = False

                    cv2.putText(
                        frame,
                        f"Pinch: {'ACTIVE' if pinch_active else 'OPEN'} | Ratio: {pinch_ratio:.2f}",                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                else:
                    pinch_state = False
                    pinch_engage_count = 0
                    pinch_release_count = 0
                    pinch_previous = False
                    hand_anchor = None
                    target_anchor_position = None
                    filtered_palm = None
                    filtered_hand_size = None
                    hand_size_anchor = None
                    gripper_toggle_start_time = None
                    gripper_release_count = 0
                    joint_hand_anchor_x = None
                    joint_angle_anchor = None


                cv2.putText(
                    frame,
                    tracking_status,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    status_color,
                    2,
                )

                cv2.putText(
                    frame,
                    "Right pinch: move | Left hand: axis | R: reset | Q: exit",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                if palm_pixel is not None:
                    cv2.circle(
                        frame,
                        palm_pixel,
                        12,
                        (0, 255, 0),
                        -1,
                    )
                
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
                                if is_new_waypoint_far_enough(
                                    taught_waypoints,
                                    last_safe_target_pose,
                                ):
                                    taught_waypoints = save_manual_waypoint(
                                        rdk,
                                        robot,
                                        base_frame,
                                        taught_waypoints,
                                        last_safe_target_pose,
                                        gripper_state,
                                    )

                                    save_gesture_status = "TARGET SAVED"

                                    print("Waypoint saved using V-sign gesture.")

                                else:
                                    save_gesture_status = "TOO CLOSE TO LAST TARGET"

                                    print("Waypoint not saved: too close to previous target.")

                                # Latch prevents repeated saving while the same V sign is held
                                save_gesture_latched = True

                    else:
                        # Gesture released, so the next V sign can save another target
                        save_gesture_start_time = None
                        save_gesture_latched = False
                        save_gesture_detected = False
                        save_gesture_status = "READY"

                (gripper_current_opening, gripper_contact_latched, contact_detected
                ) = update_gripper_animation(
                    rdk,
                    gripper,
                    grip_attach_frame,
                    pick_object,
                    gripper_current_opening,
                    gripper_target_opening,
                    gripper_contact_latched,
                )

                held_object = update_gripper_pick_drop(
                    grip_attach_frame,
                    base_frame,
                    pick_object,
                    gripper_state,
                    gripper_current_opening,
                    gripper_contact_latched,
                    held_object,
                )

                real_mode_str = "REAL ROBOT: LIVE" if REAL_ROBOT_MODE else "SIMULATION ONLY"
                real_mode_color = (0, 0, 255) if REAL_ROBOT_MODE else (0, 255, 0)
                cv2.putText(
                    frame,
                    f"Status: {real_mode_str} | Gripper: {gripper_state}",
                    (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    real_mode_color,
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"Axis mode: {axis_mode} | "
                        f"Left hand: "
                        f"{'FOUND' if axis_hand_present else 'MISSING'}"
                    ),
                    (20, 255),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )

                # Automatic continuous path recording is disabled.
                # Manual waypoint teaching will save targets only when the user presses T.
                cv2.putText(
                    frame,
                    f"Gripper: {gripper_state} | Holding: {'YES' if held_object is not None else 'NO'}",
                    (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
                recording_text = (
                    f"REC: {'ON' if recording_active else 'OFF'} | "
                    f"Points: {len(recorded_path)}"
                )

                cv2.putText(
                    frame,
                    f"Gripper command: {gripper_toggle_status}",
                    (20, 345),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    recording_text,
                    (20, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255) if recording_active else (200, 200, 200),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Taught targets: {len(taught_waypoints)}",
                    (20, 195),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 200, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Save gesture: {save_gesture_status}",
                    (20, 225),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Mode gesture: {mode_gesture_status}",
                    (20, 315),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"CONTROL MODE: {control_mode}",
                    (20, 285),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0)
                    if control_mode == "CARTESIAN"
                    else (0, 165, 255),
                    2,
                )

                if control_mode == "JOINT":
                    if selected_joint is None:
                        joint_text = "Selected Joint: NONE"
                    else:
                        current_joint_values = robot.Joints().list()
                        joint_text = (
                            f"Selected Joint: J{selected_joint} | "
                            f"{current_joint_values[selected_joint - 1]:.1f} deg"
                        )

                    cv2.putText(
                        frame,
                        joint_text,
                        (20, 315),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 165, 255),
                        2,
                    )

                if control_mode == "JOINT":

                    draw_joint_selector(
                        frame,
                        joint_hovered,
                        selected_joint,
                    )

                    cv2.putText(
                        frame,
                        f"Joint selector: {joint_selector_status}",
                        (20, 390),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 165, 255),
                        2,
                    )


                cv2.imshow(
                    "uFactory 850 Real Robot Controller - Test Ready",
                    cv2.resize(frame, (640, 360))
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("r"):
                    print("Resetting robot to initial starting posture...")
                    if REAL_ROBOT_MODE:
                        robot.MoveJ(starting_robot_joints)
                    else:
                        robot.setJoints(starting_robot_joints)
                    target.setPose(starting_tcp_pose)
                    last_safe_target_pose = starting_tcp_pose

                if key == ord("s"):
                    if not recording_active:
                        recording_active = True
                        recorded_path = []
                        recording_start_time = time.perf_counter()
                        last_record_time = 0.0
                        last_record_pose = None

                        print("Path recording started.")

                    else:
                        recording_active = False

                        print(
                            f"Path recording stopped. "
                            f"Recorded points: {len(recorded_path)}"
                        )

                        if recorded_path:
                            save_recorded_path_csv(recorded_path)

                            create_robodk_program_from_recording(
                                rdk,
                                robot,
                                base_frame,
                                recorded_path,
                            )
                if key in (ord("q"), 27):
                    break

                if key == ord("r"):
                    robot.setJoints(starting_robot_joints)
                    target.setPose(original_target_pose)

                    last_safe_target_pose = original_target_pose
                    workspace_center = (
                        starting_tcp_pose.Pos()
                    )

                    pinch_state = False
                    pinch_engage_count = 0
                    pinch_release_count = 0
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
                    # True only while one continuous Cartesian pinch is active
                    cartesian_clutch_active = False

                    print("Robot and Hand_Target reset.")
                if key == ord("t"):
                    if ENABLE_MANUAL_WAYPOINT_TEACHING:
                        taught_waypoints = save_manual_waypoint(
                            rdk,
                            robot,
                            base_frame,
                            taught_waypoints,
                            last_safe_target_pose,
                            gripper_state,
                        )

                if key == ord("g"):
                    if ENABLE_MANUAL_WAYPOINT_TEACHING:
                        create_program_from_manual_waypoints(
                            rdk,
                            robot,
                            taught_waypoints,
                        )

                if key == ord("x"):
                    taught_waypoints.clear()
                    print("Cleared taught waypoints list.")
    finally:
        camera.release()
        cv2.destroyAllWindows()

        print("Webcam target-control test stopped.")


if __name__ == "__main__":
    main()