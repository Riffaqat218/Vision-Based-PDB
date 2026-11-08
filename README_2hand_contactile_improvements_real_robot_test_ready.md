# LARA 8 Cobot 2-Hand Contactile Improvements - Real Robot Test Ready

A robust, real-time, 2-hand vision-based teleoperation and gesture control system for the **Neura LARA 8** collaborative robot and **Contactile (GAL2_MASTER)** adaptive gripper within **RoboDK**, featuring advanced control improvements, gesture hysteresis, and full **real robot testing readiness**.

This system uses a standard webcam and **MediaPipe Hand Landmarker** to track both hands simultaneously:
- **Right Hand (Control Hand)**: Controls 3D Cartesian movement / Joint rotation via an FK-anchored pinch clutch with hysteresis, latched timed-hold gripper toggle, on-screen joint button selection, and gesture-driven waypoint teaching.
- **Left Hand (Axis/Mode Hand)**: Selects active Cartesian movement axes (X, Y, Z, Free 3D, Pause), switches control modes (Cartesian vs Joint) via Thumbs Up/Down, and selects individual robot joints (J1–J6).

---

## 📂 Included Scripts

| File Name | Primary Purpose | Real Robot Execution Mode (`REAL_ROBOT_MODE`) |
|---|---|---|
| [`2hand_contactile_improvments.py`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/2hand_contactile_improvments.py) | Latest feature script optimized for **RoboDK Simulation** | `False` (Simulation only) |
| [`2hand_contactile_improvements_real_robot_test_ready.py`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/2hand_contactile_improvements_real_robot_test_ready.py) | Merged feature script configured for **Real LARA 8 Robot Testing** | Toggleable (`REAL_ROBOT_MODE = True/False`) |

---

## 🌟 Key Features

- 🖐️ **Dual-Hand MediaPipe Tracking**
  - Tracks both hands in real time (~20–30 FPS) with MediaPipe `HandLandmarker`.
  - Automatically identifies hand roles (**Right Hand** for motion clutch & gestures; **Left Hand** for axis locking & mode selection).

- 🎯 **Robust Pinch Hysteresis & False-Trigger Rejection**
  - Dual-threshold state machine eliminates jitter and accidental engagements:
    - **Engage Threshold**: Pinch distance $< 0.35$ sustained for **3 consecutive frames**.
    - **Release Threshold**: Pinch distance $> 0.50$ sustained for **5 consecutive frames**.
  - Prevents rapid toggling, sensor noise, and boundary flicker during teleoperation.

- 🕹️ **FK-Anchored Cartesian Clutch (Zero-Jump Engagement)**
  - Each time a pinch is engaged, the system anchors to the robot's actual forward kinematics pose (`robot.SolveFK(robot.Joints())`).
  - Subsequent palm displacements apply strictly relative to this anchor, preventing sudden position jumps when re-engaging pinch.

- 📍 **On-Screen Interactive Joint Selector (J1–J6)**
  - Interactive on-screen UI overlay displaying tactile buttons for Joint 1 through Joint 6.
  - Dual-method selection:
    - **Visual Pointer**: Point at on-screen buttons using the index fingertip.
    - **Left-Hand Gestures**: Finger counts 1 through 6 map directly to J1–J6.
  - Active joint is highlighted with clear on-screen visual feedback.

- 🔒 **Joint Rearm Safety Guard**
  - Switching modes (Cartesian $\leftrightarrow$ Joint) or selecting a different joint automatically triggers a safety interlock.
  - Motion is locked until the operator explicitly releases the pinch and re-engages, preventing unintended robot motion upon mode transition.

- 🧭 **Dynamic Workspace Re-Centering**
  - Whenever Cartesian mode is entered or re-engaged, the 3D Cartesian bounding box dynamically re-centers around the current robot TCP pose.
  - Maximizes available manipulator travel range without false boundary clamping.

- 🎛️ **Safe Latched Gripper with Timed Fist-Hold**
  - Replaces erratic single-frame toggles with deliberate hold-duration gestures:
    - **Close Gripper**: Hold Closed Fist for **0.70 seconds**.
    - **Open Gripper**: Hold Closed Fist / Open Palm for **1.20 seconds**.
  - Gripper state remains latched in position until explicitly toggled.
  - Integrates `send_real_gripper_command()` for physical I/O or Modbus actuation.

- 🔒 **Cartesian Axis Locking (Left Hand)**
  - **Fist (0 fingers)** $\rightarrow$ **PAUSE**: Motion locked; right-hand pinch will not move the robot.
  - **1 Finger (Index)** $\rightarrow$ **X-Axis Only**: Restricts robot movement strictly to X.
  - **2 Fingers (Index + Middle)** $\rightarrow$ **Y-Axis Only**: Restricts robot movement strictly to Y.
  - **3 Fingers (Index + Middle + Ring)** $\rightarrow$ **Z-Axis Only**: Restricts robot movement strictly to Z.
  - **4 Fingers (Index to Pinky)** $\rightarrow$ **FREE XYZ**: Full unconstrained 3D motion.

- 🎛️ **Cartesian vs. Joint Control Modes**
  - **Thumbs-Up Gesture (Left Hand)** $\rightarrow$ Switches to **CARTESIAN Mode** (TCP IK motion).
  - **Thumbs-Down Gesture (Left Hand)** $\rightarrow$ Switches to **JOINT Mode** (direct joint angle manipulation).

- ✌️ **Gesture & Keyboard Waypoint Teaching + Program Generation**
  - **V-Sign Gesture (Right Hand)**: Hold for **0.5 seconds** (or press `T`) to save current TCP pose.
  - Automatic duplicate prevention rejects waypoints closer than `20.0 mm` to the previous saved pose.
  - Press `G` to auto-generate a structured RoboDK execution program (`INSTRUCTION_CALL_PROGRAM` for gripper actions).
  - Press `S` to toggle continuous trajectory recording.

- 🛡️ **Sim-to-Real Protection & Safety Architecture**
  - **Kinematic Continuity**: Selects the nearest IK solution using `SolveIK_All` to prevent joint flips.
  - **Step Limiting**: Caps maximum TCP step (`MAX_TCP_STEP_MM = 8.0 mm`) and joint step (`MAX_JOINT_STEP_DEG = 20.0°`).
  - **Real Robot Switch (`REAL_ROBOT_MODE`)**: Toggles between simulation (`robot.setJoints()`) and real robot driver execution (`robot.MoveJ()`).
  - **Simulation Pick-and-Drop Disabled**: Prevents synthetic object teleportation during real robot tests (`ENABLE_PICK_AND_DROP = False`).
  - **Real Gripper API Hook**: `send_real_gripper_command()` ready for field bus / controller I/O integration.

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
   ├── Thumbs Up/Down (Mode Switch)   ├── Pinch Hysteresis (Engage/Release)
   ├── Finger Count 0-4 (Axis Lock)  ├── Relative Cartesian Clutch (FK Anchor)
   └── Finger Count 1-6 (J1-J6)      ├── On-Screen Joint Pointer (J1-J6)
            │                         ├── Latched Fist-Hold Gripper (0.7s/1.2s)
            │                         └── V-Sign Waypoint Teaching (0.5s)
            └────────────────┬────────────────┘
                             │
                             ▼
                  Motion Controller Logic
   ┌─────────────────────────┴─────────────────────────┐
   ▼                                                   ▼
CARTESIAN Mode                                     JOINT Mode
├── Dynamic Workspace Re-centering                 ├── Enforce Joint Rearm Safety
├── Compute Palm ΔX, ΔY (depth), ΔZ               ├── Map Horizontal Drag to Δdeg
├── Apply Active Axis Lock (X, Y, Z, Free, Pause)  ├── Enforce Joint Angle Limits
├── Workspace Clamping & Step Limiting (8mm)       └── Compute Target Joint Posture
└── Nearest-Solution IK (SolveIK_All)                  │
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
| **Thumbs Up** | Any | Hold **0.5s** to switch to **CARTESIAN Mode** (re-centers workspace) |
| **Thumbs Down** | Any | Hold **0.5s** to switch to **JOINT Mode** (arms joint safety) |
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
| **Thumb-Index Pinch** | Green Palm Circle | **Motion Clutch**: Hysteresis-checked clutch (engage $<0.35$, release $>0.50$) |
| **Index Fingertip Hover** | Button Highlight | **On-Screen Joint Selector**: Hover over J1–J6 on-screen buttons |
| **Closed Fist (Hold 0.70s)** | Orange Progress Bar | **Gripper CLOSE**: Closes gripper adaptively & fires real gripper API |
| **Closed Fist / Palm (Hold 1.20s)** | Orange Progress Bar | **Gripper OPEN**: Opens gripper to 35° & fires real gripper API |
| **V-Sign (Peace Sign)** | Blue Status Text | **Teach Waypoint**: Hold **0.5s** to save current robot pose |

### Keyboard Commands

| Key | Function |
|---|---|
| `R` | Reset robot joints and `Hand_Target` back to starting posture |
| `T` | Manually teach/save current TCP pose as a target waypoint |
| `G` | Generate executable RoboDK program from taught waypoints (`Hand_Taught_Program_<timestamp>`) |
| `S` | Toggle continuous path recording ON/OFF (saves CSV & creates RoboDK program) |
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
   - Run [`2hand_contactile_improvements_real_robot_test_ready.py`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/2hand_contactile_improvements_real_robot_test_ready.py) in simulation.
   - Verify that right-hand pinching moves `Hand_Target` smoothly with zero jump on engagement.
   - Test left-hand axis locking (`X`, `Y`, `Z`, `FREE`, `PAUSE`) and verify axis restriction in the RoboDK station view.
   - Test mode switching to `JOINT` using Thumbs Up/Down, test on-screen joint selector buttons (J1–J6), and test joint rotation with joint rearm safety.
   - Test gripper close/open with timed fist hold.

2. **Phase 2: RoboDK Driver Connection**
   - Connect RoboDK to the physical Neura LARA 8 controller via the official RoboDK robot driver or CWA/IP interface.
   - Ensure the physical robot E-stop button is in your hand.

3. **Phase 3: Enable Real Robot Execution**
   - Open [`2hand_contactile_improvements_real_robot_test_ready.py`](file:///c:/Users/rhmin/Desktop/lara/cobot_handcontroller/2hand_contactile_improvements_real_robot_test_ready.py).
   - Set `REAL_ROBOT_MODE = True` (line 58).
   - If real gripper communication is connected, set `REAL_GRIPPER_COMMANDS_ENABLED = True` (line 62).
   - Run the script and test with small, slow hand movements first.
