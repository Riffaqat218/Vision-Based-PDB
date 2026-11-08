import argparse
import unittest

import lara_contactile_mimic as MIMIC


def motion_args(**overrides):
    values = dict(
        horizontal_axis="x",
        vertical_axis="z",
        depth_axis="y",
        horizontal_scale=0.35,
        vertical_scale=0.30,
        depth_scale=0.20,
        max_horizontal=0.12,
        max_vertical=0.10,
        max_depth=0.05,
        horizontal_sign=1.0,
        vertical_sign=1.0,
        depth_sign=1.0,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class HandMappingTests(unittest.TestCase):
    def test_default_camera_axes_match_lara_setup(self):
        position = MIMIC.map_hand_delta(
            [1.0, 2.0, 3.0], 0.10, -0.10, 0.10, motion_args()
        )
        self.assertAlmostEqual(position[0], 1.035)
        self.assertAlmostEqual(position[1], 2.02)
        self.assertAlmostEqual(position[2], 3.03)

    def test_each_camera_dimension_is_clamped(self):
        position = MIMIC.map_hand_delta(
            [0.0, 0.0, 0.0], 9.0, -9.0, 9.0, motion_args()
        )
        self.assertEqual(position, [0.12, 0.05, 0.10])

    def test_sign_can_reverse_camera_horizontal_motion(self):
        position = MIMIC.map_hand_delta(
            [0.0, 0.0, 0.0], 0.10, 0.0, 0.0, motion_args(horizontal_sign=-1.0)
        )
        self.assertAlmostEqual(position[0], -0.035)

    def test_rotation_mode_maps_horizontal_hand_motion_to_tcp_yaw(self):
        args = argparse.Namespace(
            max_rotation_deg=60.0,
            rotation_scale=2.2,
            roll_scale=1.0,
        )
        quaternion = MIMIC.map_hand_rotation(
            [1.0, 0.0, 0.0, 0.0], 0.10, 0.0, 0.0, args
        )
        self.assertAlmostEqual(quaternion[0], MIMIC.math.cos(0.11))
        self.assertAlmostEqual(quaternion[3], MIMIC.math.sin(0.11))
        self.assertAlmostEqual(sum(value * value for value in quaternion), 1.0)


class FakeRobot:
    def __init__(self):
        self.pose = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
        self.servo_calls = []
        self.activated = False

    def get_tcp_pose_quaternion(self):
        return list(self.pose)

    def init_program(self):
        return None

    def activate_servo_interface(self, mode):
        self.activated = mode == "position"

    def deactivate_servo_interface(self):
        self.activated = False

    def stop(self):
        return None

    def servo_x(self, pose, velocity, acceleration, gain):
        self.pose = list(pose)
        self.servo_calls.append((list(pose), list(velocity), list(acceleration), gain))
        return 0


class LaraSafetyTests(unittest.TestCase):
    def test_all_three_camera_axes_are_enabled_by_default(self):
        args = MIMIC.parse_args([])
        self.assertEqual(
            (args.horizontal_axis, args.depth_axis, args.vertical_axis),
            ("x", "y", "z"),
        )
        self.assertTrue(args.enable_depth)

    def test_workspace_is_relative_to_pose_at_arming(self):
        args = argparse.Namespace(
            execute=True,
            robot_ip="192.0.2.1",
            workspace_x=0.05,
            workspace_y=0.12,
            workspace_z=0.10,
            target_smoothing=1.0,
            max_tcp_step=1.0,
            max_speed=1e9,
            orientation_smoothing=1.0,
            max_angular_speed=1e9,
            max_angular_step_deg=180.0,
            servo_gain=2.0,
        )
        controller = MIMIC.LaraController(args)
        controller.robot = FakeRobot()
        controller.set_armed(True)
        result = controller.send_pose([2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 0.0])
        self.assertTrue(result.workspace_limited)
        self.assertAlmostEqual(result.pose[0], 1.05)
        self.assertAlmostEqual(result.pose[1], 2.12)
        self.assertAlmostEqual(result.pose[2], 3.10)

        controller.args.max_angular_step_deg = 5.0
        target_quaternion = MIMIC.axis_angle_quaternion("z", MIMIC.math.pi / 2.0)
        rotation = controller.send_pose(result.pose[:3] + target_quaternion)
        rotated_degrees = MIMIC.math.degrees(2.0 * MIMIC.math.acos(rotation.pose[3]))
        self.assertAlmostEqual(rotated_degrees, 5.0, places=5)
        controller.close()
        self.assertFalse(controller.robot.activated)


if __name__ == "__main__":
    unittest.main()
