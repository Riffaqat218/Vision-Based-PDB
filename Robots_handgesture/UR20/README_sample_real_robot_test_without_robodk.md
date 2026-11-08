# UR20 Direct Real Robot Test (Without RoboDK)

**File**: [`sample_real_robot_test_without_robodk.py`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/UR20/sample_real_robot_test_without_robodk.py)

This script provides a **direct, standalone teleoperation interface** for the **Universal Robots UR20** cobot, inspired by the architecture in the `Mimic` folder. It communicates directly over the network via **`ur_rtde`** (Real-Time Data Exchange), completely bypassing RoboDK.

---

## 🌟 Key Architectural Features

1. **Direct Controller Connection (No RoboDK)**:
   - Uses `ur_rtde.RTDEControlInterface` and `ur_rtde.RTDEReceiveInterface` over Ethernet TCP/IP.
   - Low-latency real-time setpoint streaming (`servoL` for Cartesian, `servoJ` for Joint).
2. **Safe Dry-Run by Default**:
   - Running the script without `--execute` opens the webcam preview and tests all hand tracking and gesture logic safely without connecting to the physical robot.
3. **Zero-Jump Dynamic FK Anchoring**:
   - Whenever you engage the right-hand pinch, the script reads the live physical robot TCP pose and establishes it as the movement anchor. Releasing the pinch immediately holds the robot in place.
4. **Left-Hand Multi-Axis Safety Locks**:
   - Fist locks movement (`PAUSE`).
   - Single fingers allow isolated testing (`X`, `Y`, `Z`) before enabling full 3D translation (`FREE`).
5. **JSON Waypoint Recording**:
   - Holding a **V-Sign (Peace Sign)** for 0.5s saves the live TCP pose and joint angles directly to `ur20_waypoints.json`.

---

## ⚙️ Installation & Prerequisites

1. Install vision dependencies and the direct UR driver:
   ```powershell
   pip install opencv-python mediapipe numpy ur_rtde
   ```
   *(Note: `ur_rtde` is only required when `--execute` is supplied to connect to the physical robot).*

2. Connect your computer to the UR20 controller via Ethernet cable or local network switch. Verify you can ping the robot IP (e.g. `ping 192.168.1.100`).

---

## 🚀 How to Run

### Step 1: Dry-Run / Camera Preview (No Robot Connection)
Test hand tracking, pinch responsiveness, and axis locks safely on your camera:
```powershell
python sample_real_robot_test_without_robodk.py
```
- A green on-screen HUD displays `UR20 Direct: DRY-RUN (Sim)`.
- No robot commands will be sent.

### Step 2: Live Real Robot Test
> [!CAUTION]
> **Safety Protocol**:
> 1. Clear the physical UR20 workspace.
> 2. Always hold the physical UR20 teach pendant with the Emergency Stop (E-Stop) button accessible.

Run with `--execute` and your robot's IP:
```powershell
python sample_real_robot_test_without_robodk.py --execute --robot-ip 192.168.1.100
```
- The HUD turns red: `UR20 Direct: LIVE UR20 (Physical)`.
- The controller initialises and reads live joint and TCP coordinates directly from the controller.

---

## ✋ Gesture Control Reference

| Hand | Gesture | Function |
| :--- | :--- | :--- |
| **Right** | **Thumb + Index Pinch** | **Dead-man motion clutch**: Move hand to translate the TCP or rotate selected joint. |
| **Right** | **Release Pinch** | **Hold position**: Robot freezes at current pose immediately. |
| **Right** | **V-Sign (Peace Sign)** | Hold for **0.5 seconds** to record/save current TCP pose as a waypoint in `ur20_waypoints.json`. |
| **Left** | **Fist (0 fingers)** | `PAUSE`: Disables all robot motion even if right hand pinches. |
| **Left** | **1 Finger (Index)** | `X ONLY`: Motion constrained strictly to robot Base X axis. |
| **Left** | **2 Fingers (Index+Middle)**| `Y ONLY`: Motion constrained strictly to robot Base Y axis. |
| **Left** | **3 Fingers** | `Z ONLY`: Motion constrained strictly to vertical Z axis. |
| **Left** | **Open Palm (4-5 fingers)**| `FREE`: Full 3D Cartesian translation. |
| **Left** | **Thumbs Up / Down** | Mode Switch: Hold **Thumbs Up** for `CARTESIAN` mode; hold **Thumbs Down** for `JOINT` mode. |
| **Left** | **Finger Count (1-6)** | Joint Select: In `JOINT` mode, selects active joint (J1 to J6). |

### Keyboard Controls
- **`R`**: Re-anchor live position manually.
- **`Q` or `ESC`**: Cleanly terminate the control loop, stop the robot, and release the camera.

---

## 🔧 Useful Command-Line Parameters

```powershell
python sample_real_robot_test_without_robodk.py --help
```

Common overrides:
```text
--execute                     Enables live robot commanding
--robot-ip 192.168.1.100      UR20 controller IP
--camera-index 0              Select camera index
--workspace-x 0.25            Max displacement in X (meters)
--workspace-y 0.25            Max displacement in Y (meters)
--workspace-z 0.20            Max displacement in Z (meters)
--max-speed 0.05              Conservative speed cap (0.05 m/s)
--max-accel 0.20              Conservative acceleration limit (m/s^2)
--smoothing 0.30              Low-pass filter factor (0.1 = very smooth, 0.8 = fast)
```
