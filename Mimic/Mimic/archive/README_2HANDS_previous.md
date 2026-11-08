# LARA + Contactile Mimic: Previous Two-Hand Notes

This is the archived handover note for the two-hand version. For current start
commands and the authoritative demo status, use [`../README.md`](../README.md).

The two-hand implementation is kept separately in
`../lara_contactile_mimic_2hands.py`. It adopts the gestures from
`Mimic_Lara_with_2hands.rar`, but controls the hardware directly:

- LARA through `neurapy` (`servo_x` for the TCP and `servo_j` for joints)
- Contactile through `contactile_gripper_lib`
- no RoboDK, RobotiQ placeholder, or Windows `.venv`

The one-hand version remains available as `../lara_contactile_mimic.py`.

## Start

In camera/dry-run mode, neither LARA nor Contactile is connected:

```bash
cd ~/neura/src/johannes/Mimic
source ~/neura/.venv310/bin/activate
python3 lara_contactile_mimic_2hands.py
```

Enable real hardware only after a successful camera test:

```bash
python3 lara_contactile_mimic_2hands.py \
  --execute \
  --robot-ip 192.168.2.13 \
  --gripper-port auto
```

The safe initial state is `CARTESIAN + PAUSE`. The robot moves only after the
left hand selects an axis and the right hand closes the pinch. The left hand
may then leave the camera image; only the right control hand acts as the
dead-man. This prevents brief two-hand tracking failures from constantly
restarting the servo interface. Use `--require-axis-hand` to restore the older,
stricter requirement that both hands remain visible. Keep the emergency stop
within reach.

## Gestures

| Hand | Gesture | Function |
|---|---|---|
| Left | Fist | CARTESIAN: PAUSE; JOINT: J6 |
| Left | 1 / 2 / 3 fingers | CARTESIAN: X / Y / Z; JOINT: J1 / J2 / J3 |
| Left | 4 fingers, thumb folded | CARTESIAN: FREE XYZ; JOINT: J4 |
| Left | Open hand including thumb | JOINT: J5 |
| Left | Thumb up for 0.5 s | CARTESIAN mode |
| Left | Thumb down for 0.5 s | JOINT mode |
| Right | Thumb/index pinch | Move the TCP or selected joint |
| Right | Open hand / fist | Move Contactile to 160 mm / 0 mm |
| Right | V sign for 0.5 s | Save TCP, joints, and gripper state as a JSON waypoint |

Joint movement is limited by default to `0.03 rad/s` (about `1.7 deg/s`),
`0.12 rad/s^2` internal acceleration, and `0.5 degrees` per servo command. The
targets are smoothed during acceleration and deceleration. For an even slower
test, start with an option such as `--max-joint-speed 0.015`.

The camera updates the hand target at up to 30 Hz. Ruckig turns this into a
jerk-limited trajectory with consistent position, velocity, and acceleration.
A background loop sends it to `servo_j` at 100 Hz so JSON-RPC calls do not
block the camera and hand tracking. Joint motion stops if no fresh hand target
arrives for 0.2 seconds.

LARA documents a 1 ms cycle (1000 Hz) for direct `servo_j` control. The
installed NeuraPy JSON-RPC API opens a new TCP connection for each target and
reaches only about 200 to 230 Hz on the current controller. Requesting 1000 Hz
therefore blocks the camera without reaching the required rate. A genuinely
smooth JOINT mode requires a controller-compatible real-time interface.
CARTESIAN control through `servo_x` is not affected by this particular issue.

## Known problems and investigation ideas

### JOINT control

Direct joint control is not production-ready. Joints can vibrate noticeably
during motion. Reducing speed and acceleration and adding a Ruckig profile did
not eliminate the problem. The background rate is about 100 Hz; tests requesting
1000 Hz reached only about 200 to 230 Hz because of individual JSON-RPC/TCP
calls, while also making the camera image stutter. In addition, `servo_j`
currently returns `None` although the API documentation describes a numeric
error code. The installed client and robot server versions differ: NeuraPy
`v5.1.3`, server `v5.0.8`.

Ideas for a later solution:

- Obtain the matching NeuraPy `v5.0.8` package, or update the controller and
  client to the same version.
- Use a NEURA-supported real-time/streaming interface or a controller-side
  bridge that can feed `servo_j` at a true 1 ms rate.
- Log `get_servo_trajectory_scaling_factor()` and numeric servo return codes,
  and stop immediately on invalid responses.
- Run a reproducible hardware test with a small fixed Ruckig trajectory and no
  camera to isolate hand tracking from servo-control behavior.

Until this is resolved, use JOINT only for cautious testing and keep normal
operation in CARTESIAN with `servo_x`.

### Switching from JOINT to CARTESIAN

The switch from `servo_j` back to `servo_x` is also not reliable. Possible
causes include a joint background loop that is still running, a joint state
that has not fully settled, a servo interface left in the previous state, or
stale TCP/hand anchors after the mode change.

Suggested robust switching state machine:

1. Require the right pinch to be released and block new motion.
2. Decelerate the joint trajectory to zero velocity and acceleration, and
   confirm that the background loop has stopped.
3. Check the servo return value; if necessary, deactivate and reactivate the
   position-servo interface.
4. Reread the current joint position and TCP pose from the robot.
5. Completely reset the hand filter, joint anchor, and TCP anchor.
6. Start CARTESIAN only after explicit rearming and a new pinch.

Until this is implemented, release the pinch, stop LARA, and restart the script
in CARTESIAN after leaving JOINT mode.

## Keyboard controls and stopping

| Input | Key | Function |
|---|---|---|
| Keyboard | `T` | Save a waypoint manually |
| Keyboard | `S` or Space | Deactivate the servo interface and stop LARA |
| Keyboard | `A` | Rearm after a software stop |
| Keyboard | `Q`, `Esc`, `Ctrl+C`, or `Ctrl+Z` | Stop and exit cleanly |

After a mode change, release the right pinch once. Losing a required hand
deactivates the servo interface after the timeout. A software stop remains
latched and is not automatically cleared when hands are detected again. To
rearm, make the right hand visible, release its pinch, and press `A`. With
`--require-axis-hand`, the left hand must also be visible.

`Ctrl+Z` is intercepted and cleanly exits instead of suspending the process.
This stops LARA, deactivates the servo interface, and releases the camera.

## Directions and limits

The default mapping matches the one-hand version:

```text
Camera left/right   -> LARA X
Hand closer/farther -> LARA Y
Camera up/down      -> LARA Z
```

For example, reverse an axis with `--horizontal-sign -1`. Disable monocular
depth estimation with `--disable-depth`.

In addition to TCP workspace, speed, and step limits, joint control defaults to:

- at most 45 degrees from the current pinch anchor;
- at most 60 degrees from the joint position at activation;
- at most 0.5 degrees per update and 0.03 rad/s.

Show all options with:

```bash
python3 lara_contactile_mimic_2hands.py --help
```
