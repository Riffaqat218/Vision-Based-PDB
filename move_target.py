import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from robodk.robolink import (
    Robolink,
    ITEM_TYPE_FRAME,
    ITEM_TYPE_TARGET,
)


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
X_SCALE_MM = 500.0
Z_SCALE_MM = 400.0

# Lower value = smoother but slower response
SMOOTHING_ALPHA = 0.20

# Pinch distance divided by palm width
PINCH_THRESHOLD = 0.38

# Minimum hand movement before updating the target
DEAD_ZONE = 0.004


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
    Calculate a stable palm centre using:
    wrist + four finger base landmarks.
    """
    palm_indices = [0, 5, 9, 13, 17]

    x = sum(landmarks[index].x for index in palm_indices) / len(palm_indices)
    y = sum(landmarks[index].y for index in palm_indices) / len(palm_indices)

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
    """Connect to RoboDK and retrieve the required station items."""
    rdk = Robolink()

    target = rdk.Item(TARGET_NAME, ITEM_TYPE_TARGET)
    base_frame = rdk.Item(BASE_FRAME_NAME, ITEM_TYPE_FRAME)

    if not target.Valid():
        raise RuntimeError(
            f'RoboDK target "{TARGET_NAME}" was not found.'
        )

    if not base_frame.Valid():
        raise RuntimeError(
            f'RoboDK reference frame "{BASE_FRAME_NAME}" was not found.'
        )

    # Keep the target fixed in the station while changing its parent.
    if not target.Parent().equals(base_frame):
        print(
            f'Moving "{TARGET_NAME}" under '
            f'"{BASE_FRAME_NAME}" while preserving its position.'
        )
        target.setParentStatic(base_frame)

    target.setAsCartesianTarget()

    return rdk, target


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model was not found: {MODEL_PATH.resolve()}"
        )

    # --------------------------------------------------------
    # Connect to RoboDK
    # --------------------------------------------------------

    rdk, target = connect_to_robodk()

    original_target_pose = target.Pose()

    print("Connected to RoboDK.")
    print(f'Controlling target: "{TARGET_NAME}"')
    print("The robot will not move in this test.")

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

                    thumb_index_distance = distance_2d(
                        landmarks[4],
                        landmarks[8],
                    )

                    pinch_ratio = (
                        thumb_index_distance
                        / max(palm_width, 0.0001)
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
                            movement_z = (
                                -delta_y_normalized * Z_SCALE_MM
                            )

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

                            command_position = [
                                target_anchor_position[0] + movement_x,
                                target_anchor_position[1],
                                target_anchor_position[2] + movement_z,
                            ]

                            command_pose = target.Pose()
                            command_pose.setPos(command_position)

                            target.setPose(command_pose)

                            cv2.putText(
                                frame,
                                (
                                    f"RoboDK movement: "
                                    f"X={movement_x:+.1f} mm  "
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
                    target.setPose(original_target_pose)

                    pinch_previous = False
                    hand_anchor = None
                    target_anchor_position = None

                    print("Hand_Target reset to its original position.")

    finally:
        camera.release()
        cv2.destroyAllWindows()

        print("Webcam target-control test stopped.")


if __name__ == "__main__":
    main()