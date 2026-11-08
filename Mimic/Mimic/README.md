# LARA + Contactile Mimic Demo

Control a NEURA LARA robot and a Contactile gripper with hand gestures from the
built-in laptop camera.

There are two clearly separated variants:

| File | Status |
|---|---|
| `lara_contactile_mimic.py` | Stable one-hand version; last verified with LARA and Contactile |
| `lara_contactile_mimic_2hands.py` | Two-hand version; CARTESIAN works, while JOINT control and mode switching are experimental |

Use the one-hand version for normal demonstrations. The two-hand version is
retained for further development. Old RoboDK files, the original RAR, and the
unusable Windows environment are stored under `archive/`.

## Quick start

Open a terminal and switch to the project directory:

```bash
cd ~/neura/src/johannes/Mimic
source ~/neura/.venv310/bin/activate
```

Install the camera dependencies once:

```bash
python3 -m pip install -r requirements.txt
```

First test only the camera and gesture recognition. This command does not
connect to the robot or gripper:

```bash
python3 lara_contactile_mimic.py
```

Start the real LARA and Contactile system:

```bash
python3 lara_contactile_mimic.py \
  --execute \
  --robot-ip 192.168.2.13 \
  --gripper-port auto \
  --workspace-x 0.03 \
  --workspace-y 0.05 \
  --workspace-z 0.05 \
  --max-speed 0.02
```

These are the last small workspace limits used. Increase them only after a
supervised test.

The controls become active automatically. No arming key is required.

| Gesture or key | Action |
|---|---|
| Thumb/index pinch and hand movement | Move the LARA TCP in XYZ |
| Thumb/middle-finger pinch and hand movement | Rotate the LARA TCP |
| Release either pinch | Hold the current TCP pose |
| Open hand without a pinch | Open Contactile fully to 160 mm |
| Closed fist without a pinch | Close Contactile fully to 0 mm |
| `Q` or `Esc` | Stop and exit |

Keep the emergency stop reachable and clear the robot workspace before using
`--execute`.

## Tutorial: one-hand version

1. Connect the laptop camera, LARA network, and Contactile USB.
2. Close the Contactile Visualiser and other camera/gripper programs.
3. Start the camera dry run without `--execute`.
4. Check pinch, open-hand, and fist recognition in the video.
5. End the dry run, clear the workspace, and run the hardware command above.
6. Pinch thumb and index finger and move the hand only a small amount.
7. Release the pinch: LARA must hold its current pose.

Control is armed automatically after hardware startup. Losing the hand stops
the servo interface after the timeout. Detecting the hand again reactivates it.

## Two-hand version

First test only the camera and gestures:

```bash
python3 lara_contactile_mimic_2hands.py
```

Hardware start:

```bash
python3 lara_contactile_mimic_2hands.py \
  --execute \
  --robot-ip 192.168.2.13 \
  --gripper-port auto
```

Controls:

| Hand | Gesture | Function |
|---|---|---|
| Left | Fist | CARTESIAN: PAUSE; JOINT: J6 |
| Left | 1 / 2 / 3 fingers | CARTESIAN: X / Y / Z; JOINT: J1 / J2 / J3 |
| Left | 4 fingers without thumb | CARTESIAN: FREE; JOINT: J4 |
| Left | Open hand including thumb | JOINT: J5 |
| Left | Hold thumb up | CARTESIAN mode |
| Left | Hold thumb down | JOINT mode |
| Right | Pinch | Move the TCP or selected joint |
| Right | Open hand / fist | Open / close Contactile |
| Right | Hold V sign | Save a waypoint as JSON |

After selection, the left hand may leave the image by default; the right
control hand remains the dead-man. `--require-axis-hand` requires both hands to
remain visible.

Keyboard:

| Key | Function |
|---|---|
| `S` or Space | Latched software stop and servo deactivation |
| `A` | Rearm after releasing the pinch |
| `T` | Save a waypoint manually |
| `Q`, `Esc`, `Ctrl+C`, `Ctrl+Z` | Exit cleanly |

### Known problems in the two-hand version

Direct JOINT control is not production-ready:

- Joints can vibrate noticeably during motion.
- According to the LARA documentation, `servo_j` expects a 1 ms cycle. The
  current NeuraPy JSON-RPC connection reaches only about 200 to 230 Hz; higher
  load makes the camera image stutter.
- On the current system, `servo_j` returns `None` instead of a numeric error
  code.
- NeuraPy client `v5.1.3` does not match LARA server `v5.0.8`.
- Switching from JOINT back to CARTESIAN is not yet reliable.

Until these problems are fixed, use CARTESIAN only. After a JOINT test, release
the pinch, stop LARA, and restart the script for CARTESIAN control.

Ideas for further work:

1. Obtain the matching NeuraPy `v5.0.8` client or update the controller and
   client together.
2. Use a real controller-side real-time interface for `servo_j`.
3. During a mode change, decelerate the joint trajectory to zero and wait for
   confirmed termination of the background thread.
4. Reinitialize the servo interface and reread the TCP, joint, and hand anchors.
5. Start CARTESIAN only after the pinch has been released and control has been
   explicitly rearmed.

## Required software and hardware

- Linux laptop with a built-in camera, normally camera index `0`
- Network connection to the LARA controller
- Contactile gripper connected by USB
- `neurapy` version matching the LARA controller
- Contactile manufacturer library `contactile_gripper_lib`
- Python packages from `requirements.txt`

`neurapy` and `contactile_gripper_lib` are vendor packages and are therefore
not listed as public PyPI dependencies. Install the versions provided for the
robot and gripper.

The environment retained from the original RAR under `archive/windows_venv` is
a Windows environment and must not be used on Linux. The commands above use
`~/neura/.venv310`.

## How the code works

### 1. Camera and hand tracking

OpenCV opens the built-in laptop camera and mirrors the image so that the
preview behaves like a mirror. MediaPipe's `HandLandmarker` detects one hand
and returns 21 normalized hand landmarks for each frame.

The code calculates a stable palm center from the wrist and finger-base
landmarks. It smooths that center to reduce camera noise while retaining a
responsive left/right movement.

The model file is loaded from the same folder as the script:

```text
hand_landmarker.task
```

### 2. Translation and rotation clutches

The distance between a fingertip and the thumb is divided by the corresponding
finger length. Two separate pinch gestures select the control mode:

```text
Thumb + index finger   -> translate the TCP in XYZ
Thumb + middle finger  -> rotate the TCP in roll, pitch and yaw
```

When a pinch starts, the program stores two anchors:

- the current hand position and visible palm size
- the current LARA TCP position and orientation

Further hand movement is calculated relative to those anchors. Releasing the
pinch freezes the current target. Starting a new pinch creates new anchors, so
the robot does not jump back to an old target.

In rotation mode, left/right hand movement changes yaw, up/down changes pitch,
and turning the palm in the camera image changes roll. Each rotation clutch is
limited to ±60 degrees. Orientation smoothing, angular-speed limiting and a
maximum angular step prevent sudden quaternion changes.

### 3. Three-dimensional camera mapping

The default mapping is configured for the current LARA setup:

```text
Hand left/right       -> LARA base X, sideways
Hand closer/farther   -> LARA base Y, forward/backward
Hand up/down          -> LARA base Z, up/down
```

Left/right and up/down come from the palm center in the image. A normal laptop
camera has no depth sensor, so forward/backward motion is estimated from the
visible palm size. Moving the hand closer to the camera makes the palm appear
larger.

The depth estimate is enabled by default. Disable it if it is too noisy:

```bash
python3 lara_contactile_mimic.py --execute --disable-depth
```

Reverse an incorrectly oriented axis with a sign option, for example:

```bash
python3 lara_contactile_mimic.py --execute --horizontal-sign -1
```

### 4. LARA robot control

In hardware mode, the script connects directly to LARA through `neurapy`:

```python
Robot(socket_address="192.168.2.13")
```

It reads the current quaternion TCP pose in LARA's documented
`[X,Y,Z,qw,qx,qy,qz]` format, initializes the robot program and activates the
position servo interface. Translation and rotation targets are sent through
`servo_x`. LARA's inverse kinematics turns the robot joints as required to
reach the requested TCP pose; the code does not command unsafe raw joint
angles from camera coordinates.

Every target passes through:

1. hand-input smoothing and a small dead zone;
2. camera-to-base-axis mapping;
3. movement-range clamping;
4. workspace clamping around the activation pose;
5. TCP and quaternion smoothing, linear/angular speed limiting and step limiting;
6. the LARA `servo_x` command.

The default workspace around the activation pose is:

| Axis | Range |
|---|---:|
| X, sideways | ±0.45 m |
| Y, forward/backward | ±0.40 m |
| Z, vertical | ±0.40 m |

The maximum TCP speed remains limited to `0.05 m/s`, regardless of the larger
workspace.

### 5. Contactile gripper control

The code discovers `/dev/ttyACM*`, `/dev/ttyUSB*` and
`/dev/serial/by-id/*`, connects through `contactile_gripper_lib`, clears the
error state and optionally starts the tactile data stream.

The gesture commands use position control only:

```text
OPEN  -> release(), then pc_move_to_width(160.0)
CLOSE -> release(), then pc_move_to_width(0.0)
```

No dynamic-force or fixed-force grip mode is used. Different mechanical end
positions can be selected on the command line:

```bash
python3 lara_contactile_mimic.py \
  --execute \
  --open-width 160 \
  --closed-width 0
```

The gripper does not move automatically at startup. It moves only after a
stable open-hand or fist gesture has been detected.

### 6. Gesture detection

Open/closed detection compares each fingertip's distance from the palm center
with the corresponding middle finger joint:

- four extended fingers: `OPEN`
- zero or one extended finger: `CLOSE`
- intermediate hand shape: no gripper command
- either pinch active: no gripper command because the pinch controls the robot

A gesture must be stable for several frames before it is accepted. Repeated
frames of the same gesture do not repeatedly send the same gripper command.

### 7. Stop and recovery behavior

- Real hardware is never connected without `--execute`.
- Either pinch acts as a translation/rotation dead-man clutch.
- Losing the hand for more than 0.7 seconds stops the LARA servo interface.
- Detecting the hand again automatically reinitializes and resumes control.
- Camera read failure stops LARA and exits with an error.
- `Q`, `Esc` or `Ctrl+C` stops the robot interface and closes the camera.
- The gripper is not opened automatically on exit, preventing a held object
  from being dropped unexpectedly.

Use `Q`, `Esc`, `Ctrl+C` or `Ctrl+Z` to exit. `Ctrl+Z` is intercepted by this
script and performs the same clean shutdown instead of suspending the process.

## Useful options

Show every available option:

```bash
python3 lara_contactile_mimic.py --help
```

Common overrides:

```text
--camera-index 1       use another camera
--robot-ip ADDRESS     change the LARA IP address
--gripper-port ACM0    select a Contactile serial port
--no-gripper           control only LARA
--disable-depth        disable estimated forward/backward movement
--horizontal-sign -1   reverse sideways movement
--vertical-sign -1     reverse vertical movement
--depth-sign -1        reverse forward/backward movement
--max-speed 0.03       reduce maximum TCP speed to 3 cm/s
--max-rotation-deg 30  limit one rotation clutch to 30 degrees
--max-angular-speed .3 reduce TCP rotation speed to 0.3 rad/s
```

Command-line `--workspace-*` values override the defaults. Do not reuse older,
smaller workspace arguments if the full current range is desired.

## Troubleshooting

### Camera works once but cannot be reopened

Check for a suspended test process:

```bash
jobs -l
```

If a previous run shows as `Stopped`, resume it and close it correctly:

```bash
fg %1
```

Then press `Q`, `Ctrl+C` or `Ctrl+Z`. Current versions intercept `Ctrl+Z` and
terminate cleanly; older already-suspended processes must first be resumed.

### Wrong camera selected

```bash
python3 lara_contactile_mimic.py --camera-index 1
```

### Sideways movement is reversed

```bash
python3 lara_contactile_mimic.py --execute --horizontal-sign -1
```

### Contactile is not found

Check USB power, the cable, serial permissions and whether a device exists:

```bash
ls -l /dev/ttyACM* /dev/ttyUSB* /dev/serial/by-id/*
```

Also ensure that the Contactile Visualiser or another process is not already
using the serial port.
