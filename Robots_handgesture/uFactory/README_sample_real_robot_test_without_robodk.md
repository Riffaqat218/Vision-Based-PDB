# uFactory 850 & xArm Gripper Direct Real Robot Test (Without RoboDK)

**File**: [`sample_real_robot_test_without_robodk.py`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/uFactory/sample_real_robot_test_without_robodk.py)

This script provides a **direct, standalone teleoperation interface** for the **uFactory 850** collaborative robot and **xArm Gripper**, inspired by the architecture in the `Mimic` folder. It communicates directly over the network via **`xarm-python-sdk`** (the official UFACTORY API), completely bypassing RoboDK.

---

## 🌟 Key Architectural Features

1. **Direct Controller Connection (No RoboDK)**:
   - Uses `xarm.wrapper.XArmAPI` over Ethernet TCP/IP.
   - Communicates in Servo mode (`arm.set_mode(1)`) with streaming Cartesian (`set_servo_cartesian`) and Joint (`set_servo_angle_j`) commands.
2. **Integrated Direct Gripper Control**:
   - Sends real-time position commands directly to the xArm Gripper (`arm.set_gripper_position()`) without any external bridge.
3. **Safe Dry-Run by Default**:
   - Running the script without `--execute` opens the webcam preview and tests all hand tracking and gesture logic safely without connecting to the physical robot.
4. **Zero-Jump Dynamic FK Anchoring**:
   - Engaging the right-hand pinch queries the live TCP position from the robot and sets it as the zero reference. Releasing the pinch stops motion and holds position.
5. **Left-Hand Multi-Axis Safety Locks**:
   - Fist locks movement (`PAUSE`).
   - Single fingers allow isolated testing (`X`, `Y`, `Z`) before enabling full 3D translation (`FREE`).
6. **JSON Waypoint Recording**:
   - Holding a **V-Sign (Peace Sign)** for 0.5s saves the live TCP coordinates, joint angles, and gripper state directly to `xarm_waypoints.json`.

---

## ⚙️ Installation & Prerequisites

1. Install vision dependencies and the official xArm SDK:
   ```powershell
   pip install opencv-python mediapipe numpy xarm-python-sdk
   ```
   *(Note: `xarm-python-sdk` is only required when `--execute` is supplied to connect to the physical robot).*

2. Connect your computer to the uFactory controller via Ethernet cable or local network switch. Verify you can ping the robot IP (e.g. `ping 192.168.1.200`).

---

## 🚀 How to Run

### Step 1: Dry-Run / Camera Preview (No Robot Connection)
Test hand tracking, pinch responsiveness, gripper toggle, and axis locks safely on your camera:
```powershell
python sample_real_robot_test_without_robodk.py
```
- A green on-screen HUD displays `uFactory Direct: DRY-RUN (Sim)`.
- No robot commands will be sent.

### Step 2: Live Real Robot Test
> [!CAUTION]
> **Safety Protocol**:
> 1. Clear the physical robot workspace of personnel and obstacles.
> 2. Always hold the physical uFactory Emergency Stop (E-Stop) in your hand.

Run with `--execute` and your robot's IP:
```powershell
python sample_real_robot_test_without_robodk.py --execute --robot-ip 192.168.1.200
```
- The HUD turns red: `uFactory Direct: LIVE uFACTORY 850`.
- The controller initialises servo mode and enables live motion.

---

## ✋ Gesture Control Reference

| Hand | Gesture | Function |
| :--- | :--- | :--- |
| **Right** | **Thumb + Index Pinch** | **Dead-man motion clutch**: Move hand to translate the TCP or rotate selected joint. |
| **Right** | **Release Pinch** | **Hold position**: Robot freezes at current pose immediately. |
| **Right** | **Closed Fist (0.7s hold)** | **Gripper toggle**: Toggles physical xArm Gripper between **OPEN** and **CLOSE**. |
| **Right** | **V-Sign (Peace Sign)** | Hold for **0.5 seconds** to record/save current pose as a waypoint in `xarm_waypoints.json`. |
| **Left** | **Fist (0 fingers)** | `PAUSE`: Disables all robot motion even if right hand pinches. |
| **Left** | **1 Finger (Index)** | `X ONLY`: Motion constrained strictly to robot Base X axis. |
| **Left** | **2 Fingers (Index+Middle)**| `Y ONLY`: Motion constrained strictly to robot Base Y axis. |
| **Left** | **3 Fingers** | `Z ONLY`: Motion constrained strictly to vertical Z axis. |
| **Left** | **Open Palm (4-5 fingers)**| `FREE`: Full 3D Cartesian translation. |
| **Left** | **Thumbs Up / Down** | Mode Switch: Hold **Thumbs Up** for `CARTESIAN` mode; hold **Thumbs Down** for `JOINT` mode. |
| **Left** | **Finger Count (1-6)** | Joint Select: In `JOINT` mode, selects active joint (J1 to J6). |

### Keyboard Controls
- **`R`**: Re-anchor live position manually.
- **`Q` or `ESC`**: Cleanly terminate the control loop, disable motion, and release the camera.

---

## 🔧 Useful Command-Line Parameters

```powershell
python sample_real_robot_test_without_robodk.py --help
```

Common overrides:
```text
--execute                     Enables live robot commanding
--robot-ip 192.168.1.200      uFactory controller IP
--no-gripper                   Disable xArm gripper initialization
--camera-index 0              Select camera index
--workspace-x 0.25            Max displacement in X (meters)
--workspace-y 0.25            Max displacement in Y (meters)
--workspace-z 0.20            Max displacement in Z (meters)
--max-speed 0.05              Conservative speed cap (0.05 m/s)
--smoothing 0.30              Low-pass filter factor (0.1 = very smooth, 0.8 = fast)
```
