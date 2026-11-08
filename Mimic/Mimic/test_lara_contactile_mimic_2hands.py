import argparse
import time
import unittest

import lara_contactile_mimic_2hands as MIMIC


class TwoHandMappingTests(unittest.TestCase):
    def test_axis_lock_keeps_only_selected_base_axis(self):
        anchor = [1.0, 2.0, 3.0]
        desired = [1.1, 2.2, 3.3]
        self.assertEqual(MIMIC.apply_axis_lock(anchor, desired, "X"), [1.1, 2.0, 3.0])
        self.assertEqual(MIMIC.apply_axis_lock(anchor, desired, "Y"), [1.0, 2.2, 3.0])
        self.assertEqual(MIMIC.apply_axis_lock(anchor, desired, "Z"), [1.0, 2.0, 3.3])
        self.assertEqual(MIMIC.apply_axis_lock(anchor, desired, "FREE"), desired)
        self.assertEqual(MIMIC.apply_axis_lock(anchor, desired, "PAUSE"), anchor)

    def test_joint_target_uses_radians_and_clamps_at_anchor(self):
        target = MIMIC.map_joint_target([0.0] * 6, 3, 1.0, 180.0, 45.0)
        self.assertAlmostEqual(target[2], MIMIC.math.radians(45.0))
        self.assertEqual(target[:2] + target[3:], [0.0] * 5)

    def test_two_hands_and_safe_pause_are_defaults(self):
        args = MIMIC.parse_args([])
        self.assertEqual((args.control_hand, args.axis_hand), ("Right", "Left"))
        self.assertFalse(args.require_axis_hand)
        self.assertEqual(args.axis_mode_default, "PAUSE")
        self.assertEqual(args.hand_smoothing, 0.25)
        self.assertEqual(args.update_hz, 30.0)
        self.assertEqual(args.max_joint_speed, 0.03)
        self.assertEqual(args.max_joint_acceleration, 0.12)
        self.assertEqual(args.max_joint_jerk, 0.50)
        self.assertEqual(args.max_joint_step_deg, 0.5)
        self.assertEqual(args.joint_servo_hz, 100.0)
        self.assertEqual(args.joint_command_timeout, 0.20)


class FakeRobot:
    def __init__(self):
        self.pose = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
        self.joints = [0.0] * 6
        self.servo_j_calls = []

    def get_tcp_pose_quaternion(self):
        return list(self.pose)

    def get_current_joint_angles(self):
        return list(self.joints)

    def init_program(self):
        return None

    def activate_servo_interface(self, mode):
        return None

    def deactivate_servo_interface(self):
        return None

    def stop(self):
        return None

    def servo_j(self, joints, velocity, acceleration):
        self.joints = list(joints)
        self.servo_j_calls.append((list(joints), list(velocity), list(acceleration)))
        return 0


class JointSafetyTests(unittest.TestCase):
    def test_joint_servo_limits_step_and_workspace(self):
        args = argparse.Namespace(
            execute=True,
            robot_ip="192.0.2.1",
            joint_workspace_deg=10.0,
            max_joint_step_deg=2.0,
            max_joint_speed=1e9,
            max_joint_acceleration=1e9,
            joint_smoothing=1.0,
            servo_gain=2.0,
        )
        controller = MIMIC.DirectLaraController(args)
        controller.robot = FakeRobot()
        controller.set_armed(True)
        result = controller.send_joints([1.0] + [0.0] * 5)
        self.assertAlmostEqual(result[0], MIMIC.math.radians(2.0))
        self.assertEqual(len(controller.robot.servo_j_calls), 1)
        controller.close()

    def test_joint_servo_limits_acceleration_and_immediate_hold_stops(self):
        args = argparse.Namespace(
            execute=True,
            robot_ip="192.0.2.1",
            joint_workspace_deg=10.0,
            max_joint_step_deg=2.0,
            max_joint_speed=0.10,
            max_joint_acceleration=0.35,
            max_joint_jerk=0.50,
            joint_smoothing=1.0,
            servo_gain=2.0,
        )
        controller = MIMIC.DirectLaraController(args)
        controller.robot = FakeRobot()
        controller.set_armed(True)
        controller.last_command_time -= 0.1
        controller.send_joints([1.0] + [0.0] * 5)
        _, velocity, acceleration = controller.robot.servo_j_calls[-1]
        self.assertLessEqual(abs(velocity[0]), 0.0351)
        self.assertEqual(acceleration, [0.0] * 6)
        controller.hold_mode("JOINT")
        _, stopped_velocity, _ = controller.robot.servo_j_calls[-1]
        self.assertEqual(stopped_velocity, [0.0] * 6)
        controller.close()

    def test_joint_target_uses_background_servo_stream(self):
        args = argparse.Namespace(
            execute=True,
            robot_ip="192.0.2.1",
            joint_workspace_deg=10.0,
            max_joint_step_deg=2.0,
            max_joint_speed=0.03,
            max_joint_acceleration=0.35,
            max_joint_jerk=0.50,
            joint_smoothing=1.0,
            joint_servo_hz=100.0,
            joint_command_timeout=0.20,
            servo_gain=2.0,
        )
        controller = MIMIC.DirectLaraController(args)
        controller.robot = FakeRobot()
        controller.set_armed(True)
        controller.set_joint_target([0.2] + [0.0] * 5)
        deadline = time.monotonic() + 0.5
        while len(controller.robot.servo_j_calls) < 3 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertGreaterEqual(len(controller.robot.servo_j_calls), 3)
        self.assertTrue(
            any(
                any(abs(value) > 0.0 for value in acceleration)
                for _, _, acceleration in controller.robot.servo_j_calls
            )
        )
        controller.hold_joint_motion()
        controller.stop_joint_stream()
        self.assertFalse(controller._joint_stream_thread)
        controller.close()


if __name__ == "__main__":
    unittest.main()
