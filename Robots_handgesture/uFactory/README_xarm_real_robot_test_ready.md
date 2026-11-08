# uFactory 850 & uFactory xArm Gripper - Real Robot Test Ready Guide

This guide describes how to run and test [xarm_real_robot_test_ready.py](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/xarm_real_robot_test_ready.py) with the **uFactory 850** robot and the **uFactory xArm Gripper**.

---

## Safety Protocol & Prerequisites

> [!CAUTION]
> **ALWAYS keep the physical robot Emergency Stop (E-Stop) in your hand during live testing.**
> Ensure the physical workspace around the robot is clear of obstacles and personnel before enabling live motion.

1. **Station Verification**: Ensure RoboDK is running with the station containing:
   - `uFactory 850 Base`
   - `uFactory 850`
   - `Grip_Attach_Frame`
   - `uFactory xArm Gripper`
   - `Hand_Target`
   - `Sphere D100mm`
2. **First Test in Simulation**:
   - `REAL_ROBOT_MODE = False` (default)
   - Verify all webcam gestures and RoboDK simulation responses first.
3. **Connecting to the Real Robot**:
   - In RoboDK: Right-click **uFactory 850** -> **Connect to robot**.
   - Enter the robot controller IP address.
   - Click **Connect** and verify the status says **Ready** or **Working**.
4. **Enabling Real Motion**:
   - Set `REAL_ROBOT_MODE = True` in [xarm_real_robot_test_ready.py](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/xarm_real_robot_test_ready.py#L73).

---

## Configuration Switches

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `REAL_ROBOT_MODE` | `False` | `False` = Simulation only (`setJoints`); `True` = Real live motion (`MoveJ`). |
| `REAL_GRIPPER_COMMANDS_ENABLED` | `True` | Sends `"Gripper_Open"` / `"Gripper_Close"` instructions to the robot controller. |
| `REAL_ROBOT_SPEED_MM_S` | `40.0` | Conservative linear speed cap (mm/s) for safety. |
| `REAL_ROBOT_SPEED_JOINT_DEG_S` | `15.0` | Conservative joint speed cap (deg/s). |
| `REAL_ROBOT_ACCEL_MM_S2` | `100.0` | Linear acceleration limit (mm/s²). |
| `REAL_ROBOT_ACCEL_JOINT_DEG_S2` | `30.0` | Joint acceleration limit (deg/s²). |

---

## Gesture Control Reference

### Right Hand (Movement & Gripper & Waypoint)
- **Pinch (Thumb + Index)**: Engages tracking to move the TCP (or rotate selected joint).
- **Release Pinch**: Stops robot movement and holds position.
- **Closed Fist**: Hold for 0.7s to toggle gripper between **OPEN** and **CLOSE**.
- **V-Sign (Peace Sign)**: Hold for 0.5s to teach and save a manual waypoint target in RoboDK.

### Left Hand (Axis Lock & Mode Selection)
- **Fist (0 fingers extended)**: `PAUSE` — locks all axes (robot cannot move).
- **1 Finger (Index extended)**: `X ONLY` — motion constrained to the X axis.
- **2 Fingers (Index + Middle)**: `Y ONLY` — motion constrained to the Y axis (depth).
- **3 Fingers (Index + Middle + Ring)**: `Z ONLY` — motion constrained to the Z axis (height).
- **Open Palm (4+ fingers)**: `FREE XYZ` — full 3D Cartesian tracking.
- **Thumbs Up / Down**: Hold thumbs-up for `CARTESIAN` mode; hold thumbs-down for `JOINT` mode.

### Keyboard Shortcuts
- `r`: Instantly reset robot and target to initial home starting posture.
- `s`: Start/Stop path recording to CSV.
- `q` / `ESC`: Quit program safely.

---

## How to Run

Activate the environment and start the test script:
```powershell
.\.venv\Scripts\activate
python .\xarm_real_robot_test_ready.py
```
