# LARA8 Webcam Hand Controller — Real Robot Test

This project uses a camera and hand gestures to control the Neura LARA8 in RoboDK. The current real-test version is prepared for **robot movement** and **RobotiQ gripper open/close** only. Pick/drop object attachment is disabled for real robot testing.

## Current features

- Webcam or USB camera hand tracking using MediaPipe
- Pinch gesture to enable robot movement
- Open palm gesture to open the RobotiQ gripper
- Closed fist gesture to close the RobotiQ gripper
- RoboDK inverse kinematics for LARA8 motion
- Joint-continuous IK selection using the closest joint solution
- Motion smoothing and TCP step limiting
- Safe workspace boundary around the starting TCP
- Pick/drop disabled for real robot testing
- Path recording disabled for real robot testing

## Important safety notes

Use this first in RoboDK simulation. For real robot testing:

1. Keep the physical emergency stop reachable.
2. Use the lowest safe robot speed.
3. Keep the robot workspace clear.
4. Start with small hand movements.
5. Keep `ENABLE_WEBCAM_DEPTH = False` for the first real test.
6. Keep `ENABLE_PICK_AND_DROP = False`.
7. Keep `ENABLE_PATH_RECORDING = False`.
8. Do not enable real gripper commands until the physical gripper communication method is confirmed.

## Required files

Place these files in the same project folder:

```text
cobot_handcontroller/
├── lara8_real_robot_test_ready.py
├── hand_landmarker.task
└── README.md
```

`hand_landmarker.task` is the MediaPipe hand tracking model file.

## Python environment

Recommended environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install opencv-python mediapipe numpy robodk
```

Test imports:

```bash
python -c "import cv2, mediapipe, numpy, robodk; print('OK')"
```

## RoboDK station requirements

The RoboDK station must contain these items with matching names:

```text
Neura LARA 8 Base
Neura LARA 8
Hand_Target
RobotiQ 2F-85 Gripper (Mechanism)
```

Recommended station tree:

```text
hand_motion
└── Neura LARA 8 Base
    ├── Neura LARA 8
    │   └── RobotiQ 2F-85 Gripper (Mechanism)
    └── Hand_Target
```

`Hand_Target` should be under the fixed base frame, not inside the moving robot.

## Camera placement

For reliable control:

- Use a fixed camera mount or tripod.
- Do not move the camera after tuning the axes.
- Turn off auto-framing, auto-zoom, and auto-tracking if the camera supports them.
- Keep the hand clearly visible with good lighting.
- Keep the hand fully inside the camera view during movement.
- Use `CAMERA_INDEX = 0` for the default webcam. Change to `1`, `2`, etc. if using an external camera.

Example:

```python
CAMERA_INDEX = 0
```

## Important configuration values

Before real testing, check these values in the script:

```python
REAL_ROBOT_MODE = False
REAL_GRIPPER_COMMANDS_ENABLED = False
ENABLE_WEBCAM_DEPTH = False
ENABLE_PICK_AND_DROP = False
ENABLE_PATH_RECORDING = False
```

For simulation only:

```python
REAL_ROBOT_MODE = False
```

For real robot testing, only change `REAL_ROBOT_MODE` after RoboDK is connected to the real LARA8 driver and the robot is in a safe state.

## Controls

| Gesture / Key | Function |
|---|---|
| Thumb-index pinch | Enable robot movement |
| Release pinch | Freeze robot movement |
| Open palm | Open RobotiQ gripper |
| Closed fist | Close RobotiQ gripper |
| `R` | Reset robot and target |
| `Q` or `Esc` | Exit script |

## How the movement works

The hand does not directly command robot joints. The pipeline is:

```text
Camera image
    ↓
MediaPipe hand landmarks
    ↓
Palm center and pinch detection
    ↓
Hand_Target position update
    ↓
RoboDK IK selects nearest joint solution
    ↓
LARA8 follows the target
```

The base frame defines the coordinate system. The active robot TCP is the point that follows the target.

## Real gripper command placeholder

The RoboDK RobotiQ mechanism can visually open and close in simulation. For the real gripper, the function below must be connected to the actual communication method used in the lab:

```python
def send_real_gripper_command(opening):
    pass
```

Possible real gripper command methods depend on the setup:

- Robot controller I/O
- RoboDK driver command
- Modbus
- Vendor API
- PLC signal
- Robot program call

Do not enable real gripper commands until this part is confirmed.

## Running the script

1. Open the RoboDK station.
2. Confirm the robot and gripper item names.
3. Confirm the camera is connected.
4. Activate the Python environment.
5. Run:

```bash
python lara8_real_robot_test_ready.py
```

## First real-test checklist

Before pressing run:

- Robot is in a safe home pose.
- E-stop is reachable.
- Robot speed is reduced.
- Camera is fixed.
- Hand tracking works in simulation.
- `ENABLE_WEBCAM_DEPTH = False`.
- `ENABLE_PICK_AND_DROP = False`.
- `ENABLE_PATH_RECORDING = False`.
- Real gripper communication is confirmed before enabling physical gripper commands.

## Troubleshooting

### Camera does not open

Change:

```python
CAMERA_INDEX = 0
```

to:

```python
CAMERA_INDEX = 1
```

or try `2` if multiple cameras are connected.

### Pinch is not detected

Increase:

```python
PINCH_THRESHOLD = 0.75
```

Try values such as `0.85` or `0.95` depending on camera angle and hand distance.

### Robot moves too fast

Reduce:

```python
MAX_TCP_STEP_MM = 8.0
TARGET_SMOOTHING_ALPHA = 0.28
```

Example safer values:

```python
MAX_TCP_STEP_MM = 3.0
TARGET_SMOOTHING_ALPHA = 0.15
```

### Robot movement direction is wrong

Tune the axis mapping inside the `command_position` section. For the current version, hand left/right is mapped to the RoboDK Y axis, hand up/down is mapped to Z, and webcam depth is disabled for real testing.

### Gripper opens when it should close

Swap these values:

```python
GRIPPER_OPEN_VALUE = 85.0
GRIPPER_CLOSED_VALUE = 0.0
```

For example:

```python
GRIPPER_OPEN_VALUE = 0.0
GRIPPER_CLOSED_VALUE = 85.0
```

## Disabled functions for real testing

The following features are intentionally disabled in the real-test version:

```python
ENABLE_PICK_AND_DROP = False
ENABLE_PATH_RECORDING = False
ENABLE_WEBCAM_DEPTH = False
```

Pick/drop should stay disabled until movement and gripper open/close are validated on the real robot.

## Suggested next steps

1. Test hand tracking and robot movement in simulation.
2. Test gripper open/close in simulation.
3. Connect RoboDK to the real LARA8 at low speed.
4. Test small movement only.
5. Test gripper open/close only.
6. Add the real gripper communication method.
7. Re-enable advanced features only after basic real tests are stable.
