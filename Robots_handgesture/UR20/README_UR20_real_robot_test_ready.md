# UR20 Real Robot Test-Ready Hand Controller Guide

File: `UR20_real_robot_test_ready.py`

This script is configured specifically for the **Universal Robots UR20** cobot with **no gripper** attached, treating the robot tool frame (TCP) directly as the primary reference frame.

---

## 1. Safety Configuration Switches

At the top of `UR20_real_robot_test_ready.py`:

```python
# Set to True ONLY when connected to the physical UR20
REAL_ROBOT_MODE = False

# Safe conservative speeds for initial physical testing
REAL_ROBOT_SPEED_MM_S = 40.0           # Linear speed (mm/s)
REAL_ROBOT_SPEED_JOINT_DEG_S = 15.0     # Joint speed (deg/s)
REAL_ROBOT_ACCEL_MM_S2 = 100.0         # Linear acceleration (mm/s²)
REAL_ROBOT_ACCEL_JOINT_DEG_S2 = 30.0   # Joint acceleration (deg/s²)
```

| Mode | Behavior | HUD Indicator |
| :--- | :--- | :--- |
| **`REAL_ROBOT_MODE = False`** | Updates RoboDK digital twin only via `setJoints()` | `ROBOT: SIMULATION` (Green) |
| **`REAL_ROBOT_MODE = True`** | Commands physical UR20 movements via `MoveJ()` driver calls | `ROBOT: REAL UR20 (LIVE)` (Red) |

---

## 2. Real Robot Setup & Test Procedure

### Step 1: Verify Hardware & E-Stop
- Ensure the physical UR20 workspace is completely clear of obstacles.
- **Hold the UR20 Teach Pendant with the physical E-Stop button readily accessible at all times.**

### Step 2: Connect RoboDK to the UR20
1. Open your RoboDK station with the **UR20** robot and **UR20 Base** frame.
2. Right-click the **UR20** item in the RoboDK tree -> select **Connect to robot...**
3. Input the UR20's IP address (e.g. `192.168.1.xxx`) and port (`30000`).
4. Click **Connect**. Verify the connection indicator turns green.

### Step 3: Run First in Simulation Mode
With `REAL_ROBOT_MODE = False`:
```powershell
python .\UR20_real_robot_test_ready.py
```
- Verify both hands are tracked cleanly.
- Verify pinch engagement and axis locks.
- Check that the simulated UR20 responds smoothly.

### Step 4: Live Real Robot Testing
1. In `UR20_real_robot_test_ready.py`, change:
   ```python
   REAL_ROBOT_MODE = True
   ```
2. Start the script again.
3. **Start with Single-Axis Testing**:
   - Show **1 finger** on your Left Hand (`X-axis only`). Pinch with Right Hand to move a few millimeters in X.
   - Show **2 fingers** on your Left Hand (`Y-axis only`).
   - Show **3 fingers** on your Left Hand (`Z-axis only`).
   - Show a **fist** on your Left Hand to test the `PAUSE` function.
   - Only switch to Open Palm (`FREE` mode) once single-axis motions feel predictable and comfortable.

---

## 3. Gestures Quick Reference

### Right Hand (Control & Motion)
| Gesture | Action |
| :--- | :--- |
| **Pinch (Thumb + Index)** | Moves the robot TCP in the selected axis / Free mode |
| **Open Fingers** | Stops movement clutch (robot holds position) |
| **V-Sign (Peace)** | Hold for 0.5s to record/save the current pose as a waypoint |

### Left Hand (Axis Locks & Modes)
| Gesture | Selected Mode | Notes |
| :--- | :--- | :--- |
| **Closed Fist** | `PAUSE` | Movement frozen even if right hand pinches |
| **1 Finger (Index)** | `X ONLY` | Motion confined to robot Base X axis |
| **2 Fingers (Index + Middle)** | `Y ONLY` | Motion confined to robot Base Y axis |
| **3 Fingers** | `Z ONLY` | Motion confined to vertical Z axis |
| **Open Palm (4-5 fingers)** | `FREE` | Full 3D Cartesian translation |

### Keyboard Shortcuts
- **`R`**: Reset robot to original starting joints.
- **`Q` or `ESC`**: Safely exit program and close camera.
- **`G`**: Generate automated RoboDK motion program from taught waypoints.
- **`X`**: Clear recorded waypoints list.
