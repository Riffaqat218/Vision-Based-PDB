# Universal Robots UR20 - 2-Hand Vision Controller & Real Robot Guide

A robust, real-time vision-based dual-hand teleoperation and gesture control system for the **Universal Robots UR20** collaborative robot within **RoboDK**, featuring safety guards, axis locking, joint control, and real-robot testing readiness.

In this setup, **no gripper is attached** to the UR20; the robot Tool Center Point (TCP) serves directly as the primary reference frame.

---

## 📁 Folder Contents

| File | Purpose |
| :--- | :--- |
| [`UR20_real_robot_test_ready.py`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/UR20/UR20_real_robot_test_ready.py) | **Primary test script**: Equipped with safety interlocks, conservative speed/acceleration caps, and the `REAL_ROBOT_MODE` toggle. |
| [`UR20_handgesture.py`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/UR20/UR20_handgesture.py) | Simulation & teleoperation script for RoboDK digital twin testing. |
| [`UR20_handgesture.rdk`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/UR20/UR20_handgesture.rdk) | RoboDK station containing the UR20 robot model, base frame, and target hierarchy. |
| [`hand_landmarker.task`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/UR20/hand_landmarker.task) | MediaPipe pretrained machine learning model for 21-point 3D hand tracking. |
| [`requirements.txt`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/UR20/requirements.txt) | Python dependencies required to run the scripts. |

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
Double-click and open [`UR20_handgesture.rdk`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/UR20/UR20_handgesture.rdk) in RoboDK.
Ensure the station tree contains:
- `UR20 Base` (Reference Frame)
- `UR20` (Robot Item)
- `Hand_Target` (Target under Base Frame)

### Step 2: Run in Simulation First (Default)
Run the script while keeping `REAL_ROBOT_MODE = False`:
```powershell
python UR20_real_robot_test_ready.py
```
- Verify that the webcam window displays both hands correctly.
- Test pinch engagement, axis locks, and waypoint teaching in simulation.

### Step 3: Connect to Physical UR20 & Live Testing
> [!CAUTION]
> **Safety First**:
> 1. Clear the entire physical workspace around the UR20.
> 2. Always hold the **UR20 Teach Pendant with the physical Emergency Stop (E-Stop)** button ready in hand.

1. **Connect RoboDK to the UR20**:
   - In RoboDK: Right-click the **UR20** robot item -> select **Connect to robot...**
   - Enter the robot controller IP address (e.g., `192.168.1.xxx`) and port (`30000`).
   - Click **Connect** and confirm the status turns **Ready** / **Working**.
2. **Enable Live Robot Motion**:
   - In [`UR20_real_robot_test_ready.py`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/UR20/UR20_real_robot_test_ready.py), set:
     ```python
     REAL_ROBOT_MODE = True
     ```
3. **Run Live**:
   ```powershell
   python UR20_real_robot_test_ready.py
   ```
4. **Gradual Testing Procedure**:
   - Start with **Left Hand Fist (`PAUSE`)** to verify safety interlock.
   - Show **1 finger (`X ONLY`)**: Pinch right hand to command small millimeters along X.
   - Show **2 fingers (`Y ONLY`)**, then **3 fingers (`Z ONLY`)**.
   - Only switch to **Open Palm (`FREE XYZ`)** once single-axis movement is confirmed safe and predictable.

---

## 🛡️ Safety Configuration Switches

Configured near the top of [`UR20_real_robot_test_ready.py`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/UR20/UR20_real_robot_test_ready.py):

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `REAL_ROBOT_MODE` | `False` | `False` = Digital twin simulation (`setJoints`); `True` = Real robot motion (`MoveJ`). |
| `REAL_ROBOT_SPEED_MM_S` | `40.0` | Conservative linear speed limit (mm/s). |
| `REAL_ROBOT_SPEED_JOINT_DEG_S` | `15.0` | Conservative joint speed limit (deg/s). |
| `REAL_ROBOT_ACCEL_MM_S2` | `100.0` | Linear acceleration limit (mm/s²). |
| `REAL_ROBOT_ACCEL_JOINT_DEG_S2` | `30.0` | Joint acceleration limit (deg/s²). |
| `MAX_JOINT_STEP_DEG` | `20.0` | Maximum angular step permitted per update tick. |

---

## ✋ Gestures Quick Reference

### Right Hand (Control & Motion)
| Gesture | Action |
| :--- | :--- |
| **Pinch (Thumb + Index)** | Engages motion clutch to move robot TCP (or rotate selected joint in Joint mode). |
| **Release Pinch** | Disengages motion clutch; robot holds current pose. |
| **V-Sign (Peace Sign)** | Hold for **0.5 seconds** to record/save current TCP pose as a waypoint. |

### Left Hand (Axis Locks & Modes)
| Gesture | Selected Mode | Behavior |
| :--- | :--- | :--- |
| **Closed Fist** | `PAUSE` | Motion locked; right-hand pinch will not move robot. |
| **1 Finger (Index)** | `X ONLY` | Motion strictly constrained to Base X axis. |
| **2 Fingers (Index + Middle)** | `Y ONLY` | Motion strictly constrained to Base Y axis. |
| **3 Fingers (Index + Middle + Ring)** | `Z ONLY` | Motion strictly constrained to vertical Z axis. |
| **Open Palm (4-5 fingers)** | `FREE` | Unconstrained 3D Cartesian translation. |
| **Hold Thumbs Up / Down** | Mode Toggle | Hold for 3s to switch between `CARTESIAN` and `JOINT` modes. |
| **Finger Count (1 to 6)** | Joint Select | In `JOINT` mode: selects active joint (J1 to J6) to manipulate. |

### Keyboard Shortcuts
- **`R`**: Reset robot to original home starting joints.
- **`G`**: Auto-generate a structured RoboDK execution program from taught waypoints.
- **`X`**: Clear recorded waypoints list.
- **`Q` / `ESC`**: Safely exit program and release camera.
