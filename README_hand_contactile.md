# LARA 8 Cobot Contactile Hand Controller (`hand_contactile.py`)

A real-time, vision-based teleoperation and gesture control system for the **Neura LARA 8** collaborative robot and **Contactile (GAL2_MASTER)** adaptive gripper within **RoboDK**.

This system uses a standard webcam and **MediaPipe Hand Landmarker** to track hand movements and gestures in real time, translating them into smooth 3D robot motion, adaptive gripper control with collision-based contact detection, object pick-and-place simulation, and gesture-driven waypoint teaching.

---

## 🌟 Key Features

- 🎯 **Pinch-Based Motion Clutch**
  - Uses a **Thumb-Index Pinch** gesture as a motion clutch.
  - When pinched, hand movements in 2D webcam space map to X, Y, and Z cartesian movements of the robot's Tool Center Point (TCP).
  - When pinch is released, the robot target freezes instantly in place for safe and precise multi-step positioning.

- 🖐️ **Adaptive Contactile Gripper Control**
  - **Open Palm Gesture** (4+ fingers extended) $\rightarrow$ Opens the gripper (`GRIPPER_OPEN_VALUE = 35.0°`).
  - **Closed Fist Gesture** (≤1 finger extended) $\rightarrow$ Closes the gripper adaptively.
  - **Collision-Based Contact Detection**: When closing, finger collision meshes (`link1_mech` and `link2_mech`) are checked against target objects (`Sphere D100mm`). Once contact occurs, closing pauses instantly at the last safe opening step to prevent mesh penetration.

- ✌️ **Gesture-Based Waypoint Teaching**
  - **V-Sign Gesture** (Index & Middle fingers extended and separated): Hold for **0.5 seconds** to record the current TCP target as a waypoint without needing to touch the keyboard.
  - Automatic duplicate prevention rejects targets saved too close (`< 20.0 mm`) to the previous position.

- 🛡️ **Workspace Safety & Continuous Kinematics**
  - **Joint-Continuous IK Controller**: Selects the inverse kinematics solution closest to the current robot joint configuration to eliminate joint flips and sudden axis jumps.
  - **TCP & Joint Step Limiting**: Caps maximum TCP movement per frame (`MAX_TCP_STEP_MM = 8.0 mm`) and maximum joint movement (`MAX_JOINT_STEP_DEG = 20.0°`).
  - **Safe Workspace Bounding Box**: Clamps requested TCP positions within a defined 3D box centered around the initial robot posture.

- 📦 **Object Pick & Place Simulation**
  - Automatically parent/attaches object (`Sphere D100mm`) to `Grip_Attach_Frame` when gripper contact is verified within attachment range (`150 mm`).
  - Releases object back to the robot base frame (`Neura LARA 8 Base`) upon receiving an Open Palm gesture.

- 📝 **Trajectory Recording & RoboDK Program Generation**
  - **Path Recording (`S` Key)**: Logs robot joint states and TCP coordinates to CSV (`recordings/lara8_hand_recording_YYYYMMDD_HHMMSS.csv`) and generates a RoboDK executable program (`Hand_Guided_LARA8_Path_<timestamp>`).
  - **Manual Waypoint Program Generation (`G` Key)**: Compiles all taught targets (via `T` key or V-sign gesture) into an automated RoboDK program (`Hand_Taught_Program_<timestamp>`).

---

## 🏗️ System Architecture & Data Flow

```text
 Webcam Feed (1280x720)
         │
         ▼
 MediaPipe Hand Landmarker (hand_landmarker.task)
         │
         ├──► Pinch Ratio Detection (Thumb Tip to Index Tip vs Palm Width)
         │           │
         │           ├──► Pinch Active   --> Compute Palm Delta (ΔX, ΔY depth, ΔZ)
         │           │                           │
         │           │                           ▼
         │           │                   Smooth & Limit TCP Pose
         │           │                           │
         │           │                           ▼
         │           │                   RoboDK Nearest-Joint IK Solver
         │           │                           │
         │           │                           ▼
         │           │                   Update LARA 8 Robot Position
         │           │
         │           └──► Pinch Released --> Freeze TCP Target Position
         │
         ├──► Hand Gesture Recognition
         │           │
         │           ├──► Open Palm  --> Command Gripper OPEN (1.0) & Release Object
         │           ├──► Closed Fist --> Command Gripper CLOSE (0.0) with Mesh Contact Checks
         │           └──► V-Sign     --> Hold 0.5s to Save Taught Target Waypoint
         │
         └──► Visualization Overlay (OpenCV Frame Display)
```

---

## 📋 Prerequisites & Dependencies

### Python Requirements
- **Python 3.8+**
- Required libraries:
  ```bash
  pip install opencv-python mediapipe numpy robodk
  ```

### Required Files in Workspace
Ensure the following files are present in the same directory:
```text
cobot_handcontroller/
├── hand_contactile.py       # Main control script
├── hand_landmarker.task     # MediaPipe Hand Tracking Task Model
└── recordings/              # Output directory for exported path CSVs (auto-created)
```

---

## 🤖 RoboDK Station Setup Requirements

The script connects to an active **RoboDK** station instance via the `robolink` API. Your RoboDK station tree must include items matching these exact names:

| Item Name in RoboDK | Item Type | Description |
|---|---|---|
| `Neura LARA 8 Base` | Frame | Fixed base coordinate reference frame |
| `Neura LARA 8` | Robot | Neura LARA 8 robot arm model |
| `Hand_Target` | Target | Cartesian movement target (parented to `Neura LARA 8 Base`) |
| `GAL2_MASTER` | Mechanism/Tool | Contactile Master Gripper mechanism (0° = closed, 35° = open) |
| `Grip_Attach_Frame` | Frame | Frame located between gripper fingers for object attachment |
| `link1_mech` | Item/Collision Mesh | Left gripper finger collision link |
| `link2_mech` | Item/Collision Mesh | Right gripper finger collision link |
| `Sphere D100mm` | Object | Target workpiece object for pick-and-place collision detection |

---

## 🎮 Controls & Gesture Guide

### Hand Gestures

| Gesture | Visual Indicator | Function |
|---|---|---|
| **Thumb-Index Pinch** | Green palm center circle | **Motion Clutch Active**: Drag hand to move `Hand_Target` in 3D space |
| **Open Palm** | 4+ fingers extended | **Gripper Open**: Opens Contactile gripper & releases held object |
| **Closed Fist** | ≤1 finger extended | **Gripper Close**: Closes gripper adaptively until object contact |
| **V-Sign (Peace)** | Index & Middle extended | **Teach Target**: Hold for **0.5s** to save current robot pose as a waypoint |

### Keyboard Commands

| Key | Function |
|---|---|
| `S` | Toggle continuous path recording ON/OFF (saves CSV & creates RoboDK program) |
| `T` | Teach/Save current TCP pose as a manual waypoint target |
| `G` | Generate executable RoboDK program from all taught waypoints |
| `X` | Clear all recorded taught waypoints |
| `R` | Reset robot joints and `Hand_Target` back to starting pose |
| `Q` or `Esc` | Quit the application |

---

## ⚙️ Configuration & Parameter Tuning

Key constants defined at the top of [`hand_contactile.py`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/hand_contactile.py):

### Motion Scaling & Thresholds
```python
CAMERA_INDEX = 0             # Index of webcam (0 = default, 1 = external USB cam)
X_SCALE_MM = 750.0           # Horizontal hand movement scaling to RoboDK X (mm)
Z_UP_SCALE_MM = 450.0        # Upward hand movement scaling to RoboDK +Z (mm)
Z_DOWN_SCALE_MM = 1200.0     # Downward hand movement scaling to RoboDK -Z (mm)
Y_SCALE_MM = 220.0           # Palm size scaling to depth/Y movement (mm)
PINCH_THRESHOLD = 0.75       # Pinch distance ratio (smaller = tighter pinch required)
SMOOTHING_ALPHA = 0.30       # Landmark filtering smoothing (0.01 to 1.0)
```

### Safety & Kinematics Limits
```python
MAX_TCP_STEP_MM = 8.0        # Maximum allowed TCP displacement per update frame (mm)
MAX_JOINT_STEP_DEG = 20.0    # Maximum allowed joint change per frame (deg)
ROBOT_UPDATE_INTERVAL = 0.035# Limit IK update frequency to ~20 Hz
WORKSPACE_X_POS_MM = 350.0   # Positive X workspace boundary limit (mm)
WORKSPACE_Z_POS_MM = 300.0   # Positive Z workspace boundary limit (mm)
CLAMP_TO_WORKSPACE = True    # True = clamp to bounds; False = reject movement outside
```

### Contactile Gripper Settings
```python
GRIPPER_OPEN_VALUE = 35.0    # Joint angle for fully open gripper (deg)
GRIPPER_CLOSED_VALUE = 0.0   # Joint angle for fully closed gripper (deg)
CONTACT_CLOSE_STEP = 0.01    # Adaptive closing step ratio (1% per iteration)
CONTACT_SETTLE_SEC = 0.02    # Stabilization delay for linkage solver
ATTACH_DISTANCE_MM = 150.0   # Proximity threshold for object pick attachment (mm)
```

---

## 🚀 How to Run

1. **Launch RoboDK**:
   - Open RoboDK and load your station containing Neura LARA 8 and the Contactile gripper (`GAL2_MASTER`).
   - Ensure the item names match those listed in the station setup requirements.

2. **Verify Camera Connection**:
   - Ensure your webcam or USB camera is connected.

3. **Execute the Script**:
   ```bash
   python hand_contactile.py
   ```

4. **Operate**:
   - Bring your hand into the webcam view.
   - **Pinch** thumb and index finger to grab the motion clutch and move the robot.
   - **Release pinch** to stop robot movement.
   - **Make a fist** to grasp `Sphere D100mm`.
   - **Show V-sign** for 0.5s to mark waypoints, then press `G` to build a RoboDK movement sequence program!

---

## 🛠️ Troubleshooting

- **`RuntimeError: RoboDK robot "Neura LARA 8" was not found`**:
  - Open RoboDK and check the station tree. Ensure the robot item name matches `Neura LARA 8` exactly.
- **`FileNotFoundError: MediaPipe model was not found: hand_landmarker.task`**:
  - Download `hand_landmarker.task` from MediaPipe's official site and place it in the same directory as the script.
- **Camera standard capture error / Blank window**:
  - Change `CAMERA_INDEX = 0` to `1` or `2` in `hand_contactile.py`.
- **Robot moves too fast or erratically**:
  - Reduce `MAX_TCP_STEP_MM` (e.g., to `3.0`) or lower `TARGET_SMOOTHING_ALPHA` (e.g., to `0.15`).
- **Gripper opens instead of closing**:
  - Swap `GRIPPER_OPEN_VALUE` and `GRIPPER_CLOSED_VALUE` in the script configuration.
