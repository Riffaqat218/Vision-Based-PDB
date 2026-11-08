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
)

from robodk.robomath import eye, transl


# ============================================================
# CONFIGURATION
# ============================================================


MODEL_PATH = Path("hand_landmarker.task")

CAMERA_INDEX = 0

TARGET_NAME = "Hand_Target"
BASE_FRAME_NAME = "Neura LARA 8 Base"

# Maximum target displacement from the point where pinch starts
MAX_X_MOVEMENT_MM = 250.0
MAX_Z_MOVEMENT_MM = 250.0

# Full normalized webcam width/height mapped to RoboDK movement
X_SCALE_MM = 750.0
Z_SCALE_MM = 400.0

# Lower value = smoother but slower response
SMOOTHING_ALPHA = 0.30

# Pinch distance divided by palm width
PINCH_THRESHOLD = 0.75

# Minimum hand movement before updating the target
DEAD_ZONE = 0.004

ROBOT_NAME = "Neura LARA 8"

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
RECORDED_PROGRAM_NAME = "Hand_Guided_LARA8_Path"

# ============================================================
# MANUAL WAYPOINT TEACHING
# ============================================================

ENABLE_MANUAL_WAYPOINT_TEACHING = True

TEACH_TARGET_PREFIX = "Hand_Taught_Target"
TEACH_PROGRAM_NAME = "Hand_Taught_Program"

# ============================================================
# ROBOTIQ 2F-85 GRIPPER CONTROL
# ============================================================

ENABLE_GRIPPER_GESTURES = True

# Use the exact name from your RoboDK station tree
GRIPPER_NAME = "GAL2_MASTER"

# Based on the current Contactile mechanism:
# 0 degrees  = closed
# 35 degrees = open
#If movement is reversed, swap the two values.
GRIPPER_OPEN_VALUE = 35.0
GRIPPER_CLOSED_VALUE = 0.0

# How many stable frames before accepting open/close gesture
GRIPPER_DEBOUNCE_FRAMES = 8

# Animation speed
GRIPPER_STEP = 0.08

PICK_OBJECT_NAME = "Sphere D100mm"




GRIP_ATTACH_FRAME_NAME = "Grip_Attach_Frame"

# Smaller distance so cube only attaches when really close to the fingers
ATTACH_DISTANCE_MM = 150.0

# Local cube position after attaching to grip frame
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

# Use the exact items that successfully detected the collision
CONTACT_LEFT_ITEM_NAME = "link1_mech"
CONTACT_RIGHT_ITEM_NAME = "link2_mech"

# Normalized opening step:
# 1.0 = fully open
# 0.0 = fully closed
# With a 35-degree master range, 0.01 is about 0.35 degrees
CONTACT_CLOSE_STEP = 0.01

# Allow GAL2_V2 enough time to synchronize the linkage
CONTACT_SETTLE_SEC = 0.02

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

def load_contactile_gripper(rdk):
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


def draw_hand(frame, landmarks):
    """Draw the detected hand skeleton on the webcam image."""
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
            (255, 255, 255),
            2,
        )

    for index, point in enumerate(points):
        color = (0, 255, 255)

        if index == 4:
            color = (255, 0, 255)   # thumb tip

        if index == 8:
            color = (0, 255, 0)     # index tip

        cv2.circle(frame, point, 5, color, -1)


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

    robot.setJoints(new_joints)

    return True, "ROBOT FOLLOWING HAND"

def set_contactile_opening(gripper, opening):
    """
    opening = 1.0 means fully open
    opening = 0.0 means fully closed
    """
    if gripper is None:
        return

    opening = float(np.clip(opening, 0.0, 1.0))

    joint_value = (
        GRIPPER_CLOSED_VALUE
        + opening * (GRIPPER_OPEN_VALUE - GRIPPER_CLOSED_VALUE)
    )

    gripper.setJoints([joint_value])


def check_gripper_object_contact(
    rdk,
    left_contact_item,
    right_contact_item,
    pick_object,
):
    """
    Check collision between both Contactile fingers and the object.

    Returns:
        any_contact
        left_contact
        right_contact
    """

    if pick_object is None or not pick_object.Valid():
        return False, False, False

    if left_contact_item is None or not left_contact_item.Valid():
        return False, False, False

    if right_contact_item is None or not right_contact_item.Valid():
        return False, False, False

    left_contact = (
        rdk.Collision(left_contact_item, pick_object) > 0
    )

    right_contact = (
        rdk.Collision(right_contact_item, pick_object) > 0
    )

    any_contact = left_contact or right_contact

    return any_contact, left_contact, right_contact


def update_contactile_animation(
    rdk,
    gripper,
    left_contact_item,
    right_contact_item,
    pick_object,
    current_opening,
    target_opening,
    contact_latched,
):
    """
    Open normally.

    When closing:
    1. Move by one small step.
    2. Wait for GAL2_V2 to synchronize the linkage.
    3. Check finger-object collision.
    4. If contact occurs, return to the previous safe opening.
    5. Latch the contact so closing remains stopped.
    """

    left_contact = False
    right_contact = False

    if gripper is None:
        return (
            current_opening,
            contact_latched,
            left_contact,
            right_contact,
        )

    difference = target_opening - current_opening

    # --------------------------------------------------------
    # Opening
    # --------------------------------------------------------
    if difference > 0.0:
        # Opening resets the previous contact/grasp state
        contact_latched = False

        if abs(difference) < GRIPPER_STEP:
            new_opening = target_opening
        else:
            new_opening = current_opening + GRIPPER_STEP

        new_opening = float(
            np.clip(new_opening, 0.0, 1.0)
        )

        set_contactile_opening(
            gripper,
            new_opening,
        )

        return (
            new_opening,
            contact_latched,
            left_contact,
            right_contact,
        )

    # --------------------------------------------------------
    # Contact already detected: hold current opening
    # --------------------------------------------------------
    if contact_latched:
        return (
            current_opening,
            True,
            left_contact,
            right_contact,
        )

    # --------------------------------------------------------
    # Already reached fully closed without touching anything
    # --------------------------------------------------------
    if current_opening <= target_opening:
        set_contactile_opening(
            gripper,
            target_opening,
        )

        return (
            target_opening,
            False,
            left_contact,
            right_contact,
        )

    # --------------------------------------------------------
    # Adaptive closing: test one small closing step
    # --------------------------------------------------------
    previous_safe_opening = current_opening

    test_opening = max(
        target_opening,
        current_opening - CONTACT_CLOSE_STEP,
    )

    set_contactile_opening(
        gripper,
        test_opening,
    )

    # Allow GAL2_V2 to update all follower mechanisms
    time.sleep(CONTACT_SETTLE_SEC)

    (
        contact_detected,
        left_contact,
        right_contact,
    ) = check_gripper_object_contact(
        rdk,
        left_contact_item,
        right_contact_item,
        pick_object,
    )

    if contact_detected:
        # Return to the last position before mesh penetration
        set_contactile_opening(
            gripper,
            previous_safe_opening,
        )

        time.sleep(CONTACT_SETTLE_SEC)

        print(
            "Contact detected. "
            f"Left={left_contact}, "
            f"Right={right_contact}, "
            f"Holding opening={previous_safe_opening:.3f}"
        )

        return (
            previous_safe_opening,
            True,
            left_contact,
            right_contact,
        )

    return (
        test_opening,
        False,
        left_contact,
        right_contact,
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


def detect_gripper_gesture(landmarks, pinch_active):
    """
    Open palm  -> OPEN gripper
    Closed fist -> CLOSE gripper
    Pinch is ignored because pinch is used for robot movement.
    """
    if pinch_active:
        return None

    palm_center = calculate_palm_center(landmarks)

    index_extended = is_finger_extended(landmarks, 8, 6, palm_center)
    middle_extended = is_finger_extended(landmarks, 12, 10, palm_center)
    ring_extended = is_finger_extended(landmarks, 16, 14, palm_center)
    little_extended = is_finger_extended(landmarks, 20, 18, palm_center)

    extended_count = sum([
        index_extended,
        middle_extended,
        ring_extended,
        little_extended,
    ])

    if extended_count >= 4:
        return "OPEN"

    if extended_count <= 1:
        return "CLOSE"

    return None


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
    Contactile gripper simulated pick/drop.
    """

    if grip_attach_frame is None or not grip_attach_frame.Valid():
        print("Warning: Grip_Attach_Frame is invalid.")
        return held_object

    if pick_object is None or not pick_object.Valid():
        print("Warning: Cube 50mm is invalid.")
        return held_object

    # --------------------------------------------------------
    # Release object
    # --------------------------------------------------------
    if gripper_state == "OPEN" and held_object is not None:
        # Preserve the cube's current global position when releasing.
        held_object.setParentStatic(base_frame)

        print(f"Released object: {held_object.Name()}")
        return None

    # --------------------------------------------------------
    # Grab object
    # --------------------------------------------------------
    if (
        gripper_state == "CLOSE"
        and held_object is None
        and gripper_contact_detected
    ):
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

        print(
            f"Pick test | distance={distance_mm:.1f} mm | "
            f"opening={gripper_current_opening:.2f}"
        )

        if distance_mm <= ATTACH_DISTANCE_MM:
            # Make the cube a child of the gripping frame.
            pick_object.setParentStatic(grip_attach_frame)

            # Snap the cube to the defined local holding position.
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
    csv_path = RECORDINGS_FOLDER / f"lara8_hand_recording_{timestamp}.csv"

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


def create_program_from_manual_waypoints(
    rdk,
    robot,
    taught_waypoints,
):
    """
    Create a clean RoboDK program from manually saved waypoints.
    """
    if not taught_waypoints:
        print("No taught waypoints available.")
        return None

    timestamp = datetime.now().strftime("%H%M%S")
    program_name = f"{TEACH_PROGRAM_NAME}_{timestamp}"

    program = rdk.AddProgram(program_name, robot)

    print(f'Creating manual taught program: "{program_name}"')

    for index, waypoint in enumerate(taught_waypoints, start=1):
        target = waypoint["target"]

        program.MoveJ(target)

        print(
            f"Added waypoint {index}: "
            f"{target.Name()} | Gripper: {waypoint['gripper_state']}"
        )

    print(
        f'Program "{program_name}" created with '
        f"{len(taught_waypoints)} taught targets."
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
    gripper = load_contactile_gripper(rdk)
    
    pick_object = rdk.Item(PICK_OBJECT_NAME)

    if not pick_object.Valid():
        print(f'Warning: pick object "{PICK_OBJECT_NAME}" was not found.')
        pick_object = None
    else:
        print(f'Loaded pick object: "{pick_object.Name()}"')

    # ADD STEP 4 HERE
    # --------------------------------------------------------
    # Load Contactile collision items
    # --------------------------------------------------------

    left_contact_item = rdk.Item(CONTACT_LEFT_ITEM_NAME)
    right_contact_item = rdk.Item(CONTACT_RIGHT_ITEM_NAME)

    if not left_contact_item.Valid():
        raise RuntimeError(
            f'Left contact item "{CONTACT_LEFT_ITEM_NAME}" was not found.'
        )

    if not right_contact_item.Valid():
        raise RuntimeError(
            f'Right contact item "{CONTACT_RIGHT_ITEM_NAME}" was not found.'
        )

    print(
        f'Loaded contact items: '
        f'"{left_contact_item.Name()}" and '
        f'"{right_contact_item.Name()}"'
    )

    # --------------------------------------------------------
    # Load gripping attachment frame
    # --------------------------------------------------------

    grip_attach_frame = rdk.Item(
        GRIP_ATTACH_FRAME_NAME,
        ITEM_TYPE_FRAME,
    )
        
    grip_attach_frame = rdk.Item(GRIP_ATTACH_FRAME_NAME, ITEM_TYPE_FRAME)

    if not grip_attach_frame.Valid():
        raise RuntimeError(
            f'Grip attach frame "{GRIP_ATTACH_FRAME_NAME}" was not found.'
        )

    if gripper is not None:
        print("Contactile gripper gesture control is active.")
    else:
        print("Contactile gripper not found. Gesture will be detected but gripper will not move.")


    # Store the currently selected gripper TCP.
    tool_pose = robot.PoseTool()

    # Read the current LARA8 joints.
    starting_robot_joints = robot.Joints().list()

    # Calculate the current TCP pose relative to the robot base.
    starting_tcp_pose = robot.SolveFK(
        starting_robot_joints,
        tool=tool_pose,
        reference=eye(4),
    )

    workspace_center = starting_tcp_pose.Pos()

    print("Safe workspace center:",
        [round(value, 2) for value in workspace_center],
    )

    # Place Hand_Target exactly on the current TCP.
    # This prevents the robot from jumping when the script starts.
    target.setPose(starting_tcp_pose)

    original_target_pose = starting_tcp_pose
    last_safe_target_pose = starting_tcp_pose
    last_robot_update_time = 0.0

    print("Connected to RoboDK.")
    print(f'Controlling target: "{TARGET_NAME}"')
    print("The simulated LARA8 will follow Hand_Target.")
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
        num_hands=1,
        min_hand_detection_confidence=0.60,
        min_hand_presence_confidence=0.60,
        min_tracking_confidence=0.60,
    )

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
    print("  Pinch thumb and index finger: move Hand_Target")
    print("  Release pinch: freeze Hand_Target")
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

                tracking_status = "NO HAND"
                status_color = (0, 0, 255)

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]

                    draw_hand(frame, landmarks)

                    raw_palm = calculate_palm_center(landmarks)

                    if filtered_palm is None:
                        filtered_palm = raw_palm.copy()
                    else:
                        filtered_palm = (
                            SMOOTHING_ALPHA * raw_palm
                            + (1.0 - SMOOTHING_ALPHA) * filtered_palm
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
                            + (1.0 - DEPTH_SMOOTHING_ALPHA) * filtered_hand_size
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

                    pinch_active = pinch_ratio < PINCH_THRESHOLD
                    gripper_gesture = None
                    save_gesture_detected = False
                    save_gesture_status = "OFF"

                    if ENABLE_GESTURE_WAYPOINT_SAVE:
                        save_gesture_detected = detect_save_waypoint_gesture(
                            landmarks,
                            pinch_active,
                        )
                    if ENABLE_GRIPPER_GESTURES and not save_gesture_detected:
                        gripper_gesture = detect_gripper_gesture(
                            landmarks,
                            pinch_active,
                        )

                    if gripper_gesture is None:
                        gripper_candidate = None
                        gripper_candidate_count = 0

                    else:
                        if gripper_gesture == gripper_candidate:
                            gripper_candidate_count += 1
                        else:
                            gripper_candidate = gripper_gesture
                            gripper_candidate_count = 1

                        if (
                            gripper_candidate_count >= GRIPPER_DEBOUNCE_FRAMES
                            and gripper_gesture != gripper_state
                        ):
                            gripper_state = gripper_gesture

                            if gripper_state == "OPEN":
                                gripper_target_opening = 1.0
                                print("Contactile gripper command: OPEN")

                            elif gripper_state == "CLOSE":
                                gripper_target_opening = 0.0
                                print("Contactile gripper command: CLOSE")
                    

                    height, width = frame.shape[:2]

                    palm_pixel = (
                        int(filtered_palm[0] * width),
                        int(filtered_palm[1] * height),
                    )

                   

                    if pinch_active:
                        tracking_status = "PINCH ACTIVE - TARGET MOVING"
                        status_color = (0, 255, 0)

                        # Pinch has just started: create new movement anchors.
                        if not pinch_previous:
                            hand_anchor = filtered_palm.copy()
                            hand_size_anchor = filtered_hand_size
                            anchor_pose = target.Pose()
                            target_anchor_position = anchor_pose.Pos()

                            print(
                                "Movement clutch activated at target position:",
                                [
                                    round(value, 2)
                                    for value in target_anchor_position
                                ],
                            )

                        if (
                            hand_anchor is not None
                            and target_anchor_position is not None
                        ):
                            hand_delta = filtered_palm - hand_anchor

                            # Apply a small dead zone.
                            delta_x_normalized = hand_delta[0]
                            delta_y_normalized = hand_delta[1]

                            if abs(delta_x_normalized) < DEAD_ZONE:
                                delta_x_normalized = 0.0

                            if abs(delta_y_normalized) < DEAD_ZONE:
                                delta_y_normalized = 0.0

                            # Webcam right -> RoboDK +X
                            movement_x = (
                                delta_x_normalized * X_SCALE_MM
                            )

                            # Webcam upward means image Y becomes smaller.
                            # Therefore invert image Y for RoboDK +Z.
                            if delta_y_normalized < 0:
                                # Hand moves upward in image
                                movement_z = -delta_y_normalized * Z_UP_SCALE_MM
                            else:
                                # Hand moves downward in image
                                movement_z = -delta_y_normalized * Z_DOWN_SCALE_MM

                            movement_x = float(
                                np.clip(
                                    movement_x,
                                    -MAX_X_MOVEMENT_MM,
                                    MAX_X_MOVEMENT_MM,
                                )
                            )

                            movement_z = float(
                                np.clip(
                                    movement_z,
                                    -MAX_Z_MOVEMENT_MM,
                                    MAX_Z_MOVEMENT_MM,
                                )
                            )

                            movement_y = 0.0

                            if ENABLE_WEBCAM_DEPTH and hand_size_anchor is not None:
                                size_ratio = (
                                    filtered_hand_size
                                    / max(hand_size_anchor, 0.0001)
                                )

                                # size_ratio > 1 means hand appears bigger, so it is closer
                                movement_y = (size_ratio - 1.0) * Y_SCALE_MM * Y_DIRECTION

                                movement_y = float(
                                    np.clip(
                                        movement_y,
                                        -MAX_Y_MOVEMENT_MM,
                                        MAX_Y_MOVEMENT_MM,
                                    )
                                )

                            command_position = [
                                target_anchor_position[0] + movement_y,
                                target_anchor_position[1] + movement_x,
                                target_anchor_position[2] + movement_z,
                            ]

                            command_pose = target.Pose()
                            command_pose.setPos(command_position)

                            current_time = time.perf_counter()

                            # Limit RoboDK IK updates to approximately 20 Hz.
                            if current_time - last_robot_update_time >= ROBOT_UPDATE_INTERVAL:

                                safe_command_pose, workspace_limited = enforce_safe_workspace(
                                    command_pose,
                                    workspace_center,
                                )

                                if safe_command_pose is None:
                                    accepted = False
                                    robot_status = "OUTSIDE SAFE WORKSPACE"
                                    smooth_pose = last_safe_target_pose
                                else:
                                    smooth_pose = smooth_and_limit_pose(
                                        last_safe_target_pose,
                                        safe_command_pose,
                                    )

                                    accepted, robot_status = move_robot_to_target(
                                        robot,
                                        smooth_pose,
                                        tool_pose,
                                    )

                                    if workspace_limited and accepted:
                                        robot_status = "WORKSPACE LIMIT ACTIVE"

                            if accepted:
                                target.setPose(smooth_pose)
                                last_safe_target_pose = smooth_pose
                                status_color = (0, 255, 0)
                            else:
                                target.setPose(last_safe_target_pose)
                                status_color = (0, 0, 255)

                                tracking_status = robot_status
                                last_robot_update_time = current_time

                            cv2.putText(
                                frame,
                                (
                                    f"RoboDK movement: "
                                    f"X={movement_x:+.1f} mm  "
                                    f"Y={movement_y:+.1f} mm"
                                    f"Z={movement_z:+.1f} mm"
                                ),
                                (20, 75),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2,
                            )

                    else:
                        tracking_status = "HAND FOUND - PINCH TO MOVE"
                        status_color = (0, 255, 255)

                    pinch_previous = pinch_active

                    cv2.putText(
                        frame,
                        f"Pinch ratio: {pinch_ratio:.2f}",
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                else:
                    # Losing the hand immediately releases the clutch.
                    pinch_previous = False
                    hand_anchor = None
                    target_anchor_position = None
                    filtered_palm = None
                    filtered_hand_size = None
                    hand_size_anchor = None

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
                    "Pinch: move | S: record | T:teach | G:generate | R: reset | Q: exit",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                if result.hand_landmarks:
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

                (gripper_current_opening, gripper_contact_latched, left_contact,
                right_contact) = update_contactile_animation(
                    rdk,
                    gripper,
                    left_contact_item,
                    right_contact_item,
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

                cv2.imshow(
                    "LARA8 Webcam Hand Control - Target Test",
                    cv2.resize(frame, (640, 360))
                )

                key = cv2.waitKey(1) & 0xFF

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

                    pinch_previous = False
                    hand_anchor = None
                    target_anchor_position = None
                    filtered_palm = None

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