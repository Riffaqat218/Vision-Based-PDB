# LARA 8 Cobot 2-Hand Contactile Axis Lock & Joint Controller

A real-time, 2-hand vision-based teleoperation and gesture control system for the **Neura LARA 8** collaborative robot and **Contactile (GAL2_MASTER)** adaptive gripper within **RoboDK**, supporting both simulation and **real robot testing**.

This system uses a standard webcam and **MediaPipe Hand Landmarker** to track both hands simultaneously:
- **Right Hand (Control Hand)**: Controls 3D Cartesian movement / Joint rotation via pinch clutch, gripper open/close gestures, and gesture-driven waypoint saving.
- **Left Hand (Axis/Mode Hand)**: Selects active Cartesian movement axes (X, Y, Z, Free 3D, Pause), switches control modes (Cartesian vs Joint) using Thumbs Up/Down, and selects individual robot joints (J1–J6) for single-joint tuning.

---

## 📂 Included Scripts

| File Name | Primary Purpose | Real Robot Execution Mode (`REAL_ROBOT_MODE`) |
|---|---|---|
| [`2hand_contactile_axis_lock.py`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/2hand_contactile_axis_lock.py) | Full feature script optimized for **RoboDK Simulation** | `False` (Simulation only) |
| [`2hand_contactile_axis_lock_real_robot_test_ready.py`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/2hand_contactile_axis_lock_real_robot_test_ready.py) | Configured script ready for **Real LARA 8 Robot Testing** | Toggleable (`REAL_ROBOT_MODE = True/False`) |

---

## 🌟 Key Features

- 🖐️ **Dual-Hand MediaPipe Tracking**
  - Tracks both hands in real time at ~20 Hz with MediaPipe `HandLandmarker`.
  - Automatically identifies hand roles (**Right Hand** for motion clutch & gestures; **Left Hand** for axis locking & joint selection).

- 🔒 **Cartesian Axis Locking (Left Hand)**
  - **Fist (0 fingers)** $\rightarrow$ **PAUSE**: Locks motion; right-hand pinch will not move the robot.
  - **1 Finger (Index)** $\rightarrow$ **X-Axis Only**: Restricts robot movement strictly to X.
  - **2 Fingers (Index + Middle)** $\rightarrow$ **Y-Axis Only**: Restricts robot movement strictly to Y.
  - **3 Fingers (Index + Middle + Ring)** $\rightarrow$ **Z-Axis Only**: Restricts robot movement strictly to Z.
  - **4 Fingers (Index to Pinky)** $\rightarrow$ **FREE XYZ**: Full unconstrained 3D motion.

- 🎛️ **Cartesian vs. Joint Control Modes**
  - **Thumbs-Up Gesture (Left Hand)** $\rightarrow$ Switches to **CARTESIAN Mode** (TCP IK motion).
  - **Thumbs-Down Gesture (Left Hand)** $\rightarrow$ Switches to **JOINT Mode** (direct joint angle manipulation).

- 🦾 **Single Joint Selection & Control (JOINT Mode)**
  - Select robot joints 1 to 6 using Left Hand finger counts:
    - **1 Finger** $\rightarrow$ **J1** | **2 Fingers** $\rightarrow$ **J2** | **3 Fingers** $\rightarrow$ **J3**
    - **4 Fingers (Thumb folded)** $\rightarrow$ **J4** | **5 Fingers (Open Palm)** $\rightarrow$ **J5** | **Closed Fist** $\rightarrow$ **J6**
  - Pinching with the Right Hand and moving horizontally rotates the selected joint smoothly up to $\pm 45^\circ$ from its starting posture.

- 🖐️ **Adaptive Contactile Gripper Control**
  - **Open Palm Gesture (Right Hand)** $\rightarrow$ Opens the Contactile gripper (`35.0°`).
  - **Closed Fist Gesture (Right Hand)** $\rightarrow$ Closes the Contactile gripper adaptively (`0.0°`).
  - **Collision-Based Contact Detection**: Checks collision meshes (`link1_mech` and `link2_mech`) against target objects (`Sphere D100mm`) in simulation to prevent mesh overlap.

- ✌️ **Gesture-Based Waypoint Teaching**
  - **V-Sign Gesture (Right Hand)**: Hold for **0.5 seconds** to save the current robot TCP pose as a target waypoint.
  - Automatic duplicate prevention rejects waypoints saved closer than `20.0 mm` to the previous saved pose.

- 🛡️ **Real Robot Readiness & Safety Protection**
  - **Kinematic Continuity**: Selects the nearest IK solution using `SolveIK_All` to prevent joint flips and axis jumps.
  - **Joint & TCP Step Limiting**: Caps maximum TCP step (`MAX_TCP_STEP_MM = 8.0 mm`) and joint step (`MAX_JOINT_STEP_DEG = 20.0°`).
  - **Workspace Clamping**: Restricts robot movements within a safe 3D bounding box centered on the starting TCP pose.
  - **Real Robot Switch (`REAL_ROBOT_MODE`)**: Toggles between simulation (`robot.setJoints()`) and real robot driver execution (`robot.MoveJ()`).
  - **Real Gripper API Hook**: `send_real_gripper_command()` for connecting Modbus TCP / URCap / controller I/O signals.

---

## 🏗️ System Architecture & Data Flow

```text
               Webcam Video Stream (1280x720)
                             │
                             ▼
             MediaPipe HandLandmarker (num_hands=2)
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   Left Hand (Axis / Mode)           Right Hand (Control / Gripper)
   ├── Thumbs Up/Down --> CARTESIAN / JOINT Mode
   ├── Finger Count 0-4 --> Axis Lock (PAUSE/X/Y/Z/FREE)
   └── Finger Count 1-6 --> Joint Select (J1..J6)
            │                                 │
            └────────────────┬────────────────┘
                             │
                             ▼
                  Motion Controller Logic
   ┌─────────────────────────┴─────────────────────────┐
   ▼                                                   ▼
CARTESIAN Mode                                     JOINT Mode
├── Compute Palm ΔX, ΔY (depth), ΔZ               ├── Map Horizontal Drag
├── Apply Active Axis Lock                         │   to Selected Joint Δdeg
├── Workspace Clamping & Step Limiting             ├── Enforce Joint Limits
└── Nearest-Solution IK (SolveIK_All)              └── Update Joint Angle
   │                                                   │
   └─────────────────────────┬─────────────────────────┘
                             │
                             ▼
             Target Execution Mode Check
             ├── REAL_ROBOT_MODE = False  --> robot.setJoints() [Simulation]
             └── REAL_ROBOT_MODE = True   --> robot.MoveJ()     [Real Robot]
```

---

## 🎮 Gestures & Controls Quick Reference

### Left Hand Gestures (Axis & Mode Selection)

| Gesture | Mode Active | Action / Function |
|---|---|---|
| **Thumbs Up** | Any | Hold **0.5s** to switch to **CARTESIAN Mode** |
| **Thumbs Down** | Any | Hold **0.5s** to switch to **JOINT Mode** |
| **Closed Fist (0 fingers)** | CARTESIAN | **PAUSE**: Motion locked; right pinch is disarmed |
| **1 Finger (Index)** | CARTESIAN | Lock movement to **X-Axis only** |
| **2 Fingers (Index+Middle)** | CARTESIAN | Lock movement to **Y-Axis only** |
| **3 Fingers (Index+Middle+Ring)** | CARTESIAN | Lock movement to **Z-Axis only** |
| **4 Fingers (Four fingers)** | CARTESIAN | **FREE 3D**: Full X, Y, and Z motion |
| **1 Finger** | JOINT | Select **Joint 1 (J1)** |
| **2 Fingers** | JOINT | Select **Joint 2 (J2)** |
| **3 Fingers** | JOINT | Select **Joint 3 (J3)** |
| **4 Fingers (Thumb folded)** | JOINT | Select **Joint 4 (J4)** |
| **5 Fingers (Open Palm)** | JOINT | Select **Joint 5 (J5)** |
| **Closed Fist** | JOINT | Select **Joint 6 (J6)** |

### Right Hand Gestures (Control & Operations)

| Gesture | Visual Feedback | Function |
|---|---|---|
| **Thumb-Index Pinch** | Green Palm Circle | **Motion Clutch**: Move TCP (Cartesian) or rotate Joint (Joint mode) |
| **Open Palm** | Yellow Text | **Gripper OPEN**: Opens Contactile gripper & calls real gripper API |
| **Closed Fist** | Yellow Text | **Gripper CLOSE**: Closes Contactile gripper & calls real gripper API |
| **V-Sign (Peace Sign)** | Blue Status Text | **Teach Waypoint**: Hold **0.5s** to save current robot pose |

### Keyboard Commands

| Key | Function |
|---|---|
| `R` | Reset robot joints and `Hand_Target` back to starting posture |
| `T` | Manually teach/save current TCP pose as a target waypoint |
| `G` | Generate executable RoboDK program from all taught waypoints (`Hand_Taught_Program_<timestamp>`) |
| `S` | Toggle path recording ON/OFF (saves CSV & creates RoboDK program) |
| `X` | Clear all saved waypoints from memory |
| `Q` or `Esc` | Quit the application cleanly |

---

## 🤖 RoboDK Station Setup Requirements

Your RoboDK station tree must contain items with the following exact names:

| Item Name in RoboDK | Item Type | Function |
|---|---|---|
| `Neura LARA 8 Base` | Frame | Fixed robot base reference frame |
| `Neura LARA 8` | Robot | Neura LARA 8 robot arm model |
| `Hand_Target` | Target | Motion target parented to `Neura LARA 8 Base` |
| `GAL2_MASTER` | Mechanism | Contactile Master Gripper mechanism (0° closed, 35° open) |
| `link1_mech` | Item/Mesh | Left gripper finger collision link |
| `link2_mech` | Item/Mesh | Right gripper finger collision link |
| `Grip_Attach_Frame` | Frame | Gripper center attachment frame |
| `Sphere D100mm` | Object | Target object for collision and pick-and-place testing |

---

## 🚨 Real Robot Testing Step-by-Step Procedure

1. **Phase 1: Simulation Verification (`REAL_ROBOT_MODE = False`)**
   - Run [`2hand_contactile_axis_lock_real_robot_test_ready.py`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/2hand_contactile_axis_lock_real_robot_test_ready.py) in simulation.
   - Verify that right-hand pinching moves `Hand_Target` smoothly.
   - Test left-hand axis locking (`X`, `Y`, `Z`, `FREE`, `PAUSE`) and verify axis restriction in the RoboDK station view.
   - Test mode switching to `JOINT` using Thumbs Up/Down and test joint rotations (`J1`–`J6`).

2. **Phase 2: RoboDK Driver Connection**
   - Connect RoboDK to the physical Neura LARA 8 controller via the official RoboDK robot driver or CWA/IP interface.
   - Ensure the physical robot E-stop button is in your hand.

3. **Phase 3: Enable Real Robot Execution**
   - Open [`2hand_contactile_axis_lock_real_robot_test_ready.py`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/2hand_contactile_axis_lock_real_robot_test_ready.py).
   - Change `REAL_ROBOT_MODE = True`.
   - If real gripper communication is connected, change `REAL_GRIPPER_COMMANDS_ENABLED = True`.
   - Run the script and test with small, slow hand movements first.
