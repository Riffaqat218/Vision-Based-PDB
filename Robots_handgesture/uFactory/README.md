# uFactory 850 & xArm Gripper - 2-Hand Vision Controller & Real Robot Guide

A robust, real-time vision-based dual-hand teleoperation and gesture control system for the **uFactory 850** collaborative robot and **uFactory xArm Gripper** within **RoboDK**, featuring safety guards, gripper toggle, axis locking, joint control, and real-robot testing readiness.

---

## 📁 Folder Contents

| File | Purpose |
| :--- | :--- |
| [`xarm_real_robot_test_ready.py`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/uFactory/xarm_real_robot_test_ready.py) | **Primary test script**: Equipped with safety interlocks, gripper controls, conservative speed caps, and the `REAL_ROBOT_MODE` toggle. |
| [`xarm_handgesture.py`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/uFactory/xarm_handgesture.py) | Simulation & teleoperation script for RoboDK digital twin testing. |
| [`xarm_handgesture.rdk`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/uFactory/xarm_handgesture.rdk) | RoboDK station containing the uFactory 850 robot, xArm Gripper mechanism, attachment frames, and target hierarchy. |
| [`hand_landmarker.task`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/uFactory/hand_landmarker.task) | MediaPipe pretrained machine learning model for 21-point 3D hand tracking. |
| [`requirements.txt`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/uFactory/requirements.txt) | Python dependencies required to run the scripts. |

---

## ⚙️ Installation & Prerequisites

1. **Python Environment**:
   Python 3.10+ is recommended. Activate your virtual environment:
   ```powershell
   # From workspace root or folder:
   ..\.venv\Scripts\activate
   ```
2. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
3. **RoboDK Software**:
   Make sure [RoboDK](https://robodk.com/) is installed and running on your computer.

---

## 🚀 Quick Start Guide

### Step 1: Open the RoboDK Station
Double-click and open [`xarm_handgesture.rdk`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/uFactory/xarm_handgesture.rdk) in RoboDK.
Ensure the station tree contains:
- `uFactory 850 Base` (Reference Frame)
- `uFactory 850` (Robot Item)
- `Grip_Attach_Frame`
- `uFactory xArm Gripper` (Gripper Mechanism)
- `Hand_Target` (Target under Base Frame)

### Step 2: Run in Simulation First (Default)
Run the script while keeping `REAL_ROBOT_MODE = False`:
```powershell
python xarm_real_robot_test_ready.py
```
- Verify that the webcam window displays both hands correctly.
- Test pinch engagement, gripper open/close toggle, and axis locks in simulation.

### Step 3: Connect to Physical uFactory 850 & Live Testing
> [!CAUTION]
> **Safety Protocol**:
> 1. Ensure the robot workspace is entirely clear of obstacles and personnel.
> 2. Always hold the **uFactory physical Emergency Stop (E-Stop)** ready in hand.

1. **Connect RoboDK to the Robot**:
   - In RoboDK: Right-click the **uFactory 850** robot item -> select **Connect to robot...**
   - Enter the robot controller IP address.
   - Click **Connect** and confirm the status turns **Ready** / **Working**.
2. **Enable Live Robot Motion**:
   - In [`xarm_real_robot_test_ready.py`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/uFactory/xarm_real_robot_test_ready.py), set:
     ```python
     REAL_ROBOT_MODE = True
     ```
3. **Run Live**:
   ```powershell
   python xarm_real_robot_test_ready.py
   ```
4. **Gradual Testing Procedure**:
   - Test **Left Hand Fist (`PAUSE`)** first to confirm motion lock.
   - Test single-axis motions: 1 finger (`X ONLY`), 2 fingers (`Y ONLY`), 3 fingers (`Z ONLY`).
   - Test the gripper toggle using a held closed fist on your right hand.
   - Only switch to **Open Palm (`FREE XYZ`)** once single-axis motions feel predictable.

---

## 🛡️ Safety Configuration Switches

Configured near the top of [`xarm_real_robot_test_ready.py`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/uFactory/xarm_real_robot_test_ready.py):

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `REAL_ROBOT_MODE` | `False` | `False` = Digital twin simulation (`setJoints`); `True` = Real robot live motion (`MoveJ`). |
| `REAL_GRIPPER_COMMANDS_ENABLED` | `True` | Sends live gripper commands (`"Gripper_Open"` / `"Gripper_Close"`) to the robot controller. |
| `REAL_ROBOT_SPEED_MM_S` | `40.0` | Conservative linear speed limit (mm/s). |
| `REAL_ROBOT_SPEED_JOINT_DEG_S` | `15.0` | Conservative joint speed limit (deg/s). |
| `REAL_ROBOT_ACCEL_MM_S2` | `100.0` | Linear acceleration limit (mm/s²). |
| `REAL_ROBOT_ACCEL_JOINT_DEG_S2` | `30.0` | Joint acceleration limit (deg/s²). |
| `MAX_JOINT_STEP_DEG` | `20.0` | Maximum angular step permitted per update tick. |

---

## ✋ Gestures Quick Reference

### Right Hand (Control, Gripper & Waypoint)
| Gesture | Action |
| :--- | :--- |
| **Pinch (Thumb + Index)** | Engages motion clutch to move robot TCP (or rotate selected joint in Joint mode). |
| **Release Pinch** | Disengages motion clutch; robot holds current pose. |
| **Closed Fist (Timed Hold)** | Hold closed fist for **0.7 seconds** to toggle gripper state between **OPEN** and **CLOSE**. |
| **V-Sign (Peace Sign)** | Hold for **0.5 seconds** to record/save current TCP pose as a waypoint in RoboDK. |

### Left Hand (Axis Locks & Modes)
| Gesture | Selected Mode | Behavior |
| :--- | :--- | :--- |
| **Closed Fist** | `PAUSE` | Motion locked; right-hand pinch will not move robot. |
| **1 Finger (Index)** | `X ONLY` | Motion strictly constrained to Base X axis. |
| **2 Fingers (Index + Middle)** | `Y ONLY` | Motion strictly constrained to Base Y axis (depth). |
| **3 Fingers (Index + Middle + Ring)** | `Z ONLY` | Motion strictly constrained to vertical Z axis (height). |
| **Open Palm (4-5 fingers)** | `FREE XYZ` | Unconstrained 3D Cartesian translation. |
| **Thumbs Up / Down** | Mode Switch | Hold **Thumbs Up** for `CARTESIAN` mode; hold **Thumbs Down** for `JOINT` mode. |
| **Finger Count (1 to 6)** | Joint Select | In `JOINT` mode: selects active joint (J1 to J6) to manipulate. |

### Keyboard Shortcuts
- **`r`**: Instantly reset robot and target to initial home starting posture.
- **`s`**: Start / Stop continuous trajectory logging to CSV.
- **`q` / `ESC`**: Safely exit program and release camera.
