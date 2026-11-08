from robodk import robomath
from robodk import robomath
from robodk import robomath
import math
import time
from pathlib import Path
from copy import deepcopy  #smoothwere need to be imported from deepcopy module

import cv2
import mediapipe as mp
import numpy as np

from robodk.robolink import (
    Robolink,
    ITEM_TYPE_FRAME,
    ITEM_TYPE_TARGET,
    ITEM_TYPE_ROBOT,
)

from robodk.robomath import eye


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
PINCH_THRESHOLD = 0.38

# Minimum hand movement before updating the target
DEAD_ZONE = 0.004

ROBOT_NAME = "Neura LARA 8"

# Maximum joint movement allowed in one webcam update
MAX_JOINT_STEP_DEG = 4.0

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
Z_DOWN_SCALE_MM = 700.0

# Safe workspace around the starting TCP position, in mm
WORKSPACE_X_NEG_MM = 250.0
WORKSPACE_X_POS_MM = 250.0

WORKSPACE_Y_NEG_MM = 100.0
WORKSPACE_Y_POS_MM = 100.0

WORKSPACE_Z_NEG_MM = 180.0
WORKSPACE_Z_POS_MM = 250.0

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


def move_robot_to_target(robot, target_pose, tool_pose):
    """
    Solve inverse kinematics for the target TCP pose and update
    the simulated robot joints.

    Returns:
        accepted: True if the robot was updated
        status: explanation of the result
    """
    current_joints = robot.Joints().list()
    robot_dof = len(current_joints)

    # Target pose is TCP relative to the LARA8 base.
    joint_solution_matrix = robot.SolveIK(
        target_pose,
        joints_approx=current_joints,
        tool=tool_pose,
        reference=eye(4),
    )

    joint_solution = joint_solution_matrix.list()

    # No IK solution means the TCP target is unreachable.
    if len(joint_solution) < robot_dof:
        return False, "IK FAILED - TARGET UNREACHABLE"

    # Some robots may return extra values. Keep only robot axes.
    joint_solution = joint_solution[:robot_dof]

    joint_changes = [
        abs(wrapped_joint_difference(new_joint, current_joint))
        for new_joint, current_joint
        in zip(joint_solution, current_joints)
    ]

    maximum_joint_change = max(joint_changes)

    if maximum_joint_change > MAX_JOINT_STEP_DEG:
        return (
            False,
            f"JOINT JUMP REJECTED: {maximum_joint_change:.1f} DEG",
        )

    # Immediate simulated joint update.
    robot.setJoints(joint_solution)

    return True, "ROBOT FOLLOWING HAND"


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

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model was not found: {MODEL_PATH.resolve()}"
        )

    # --------------------------------------------------------
    # Connect to RoboDK
    # --------------------------------------------------------

    rdk, robot, target, base_frame = connect_to_robodk()

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

    start_time = time.perf_counter()
    previous_timestamp_ms = -1

    print()
    print("Controls:")
    print("  Pinch thumb and index finger: move Hand_Target")
    print("  Release pinch: freeze Hand_Target")
    print("  R: reset target")
    print("  Q or Esc: exit")

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

                    pinch_ratio = (
                        thumb_index_distance
                        / max(filtered_hand_size, 0.0001)
                    )

                    pinch_active = pinch_ratio < PINCH_THRESHOLD

                    height, width = frame.shape[:2]

                    palm_pixel = (
                        int(filtered_palm[0] * width),
                        int(filtered_palm[1] * height),
                    )

                    cv2.circle(
                        frame,
                        palm_pixel,
                        12,
                        (0, 255, 0),
                        -1,
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
                                target_anchor_position[0] + movement_x,
                                target_anchor_position[1] + movement_y,
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
                                    f"Z={movement_z:+.1f} mm"
                                    f"Y={movement_y:+.1f} mm"
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
                    "Pinch: move | Release: freeze | R: reset | Q: exit",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(
                    "LARA8 Webcam Hand Control - Target Test",
                    cv2.resize(frame, (640, 360))
                )

                key = cv2.waitKey(1) & 0xFF

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

    finally:
        camera.release()
        cv2.destroyAllWindows()

        print("Webcam target-control test stopped.")


if __name__ == "__main__":
    main()