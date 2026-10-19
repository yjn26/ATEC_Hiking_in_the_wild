import unittest
import xml.etree.ElementTree as ET
import subprocess
import sys
from pathlib import Path

from tools.generate_training_urdf import TRAINING_URDF_PATH, referenced_mesh_paths
from tools.validate_asset_contract import CONTRACT_PATH, load_json
from assets.tron2a_10dof import load_training_asset_spec
from assets.official_policy import (
    load_official_policy_spec,
    make_joint_position_targets,
    runtime_joint_ids,
)
from tools.inspect_imported_robot import inspect_offline_training_asset, runtime_validation_contract


class TrainingUrdfTest(unittest.TestCase):
    def test_official_policy_actions_are_mapped_by_joint_name(self):
        training_spec = load_training_asset_spec()
        policy_spec = load_official_policy_spec()
        scrambled_runtime_names = [
            "arm1_Joint",
            "proximal_pitch_R_Joint",
            "knee_L_Joint",
            "gripper2_Joint",
            "proximal_pitch_L_Joint",
            "ankle_pitch_R_Joint",
            "proximal_roll_L_Joint",
            "arm2_Joint",
            "proximal_yaw_R_Joint",
            "ankle_pitch_L_Joint",
            "proximal_roll_R_Joint",
            "knee_R_Joint",
            "proximal_yaw_L_Joint",
            "arm3_Joint",
            "arm4_Joint",
            "arm5_Joint",
            "arm6_Joint",
            "gripper1_Joint",
        ]

        self.assertEqual(policy_spec.action_joint_names, tuple(training_spec.policy_actuators))
        self.assertEqual(
            runtime_joint_ids(scrambled_runtime_names, policy_spec.action_joint_names),
            [4, 6, 12, 2, 9, 1, 10, 8, 11, 5],
        )

    def test_policy_targets_leave_locked_joint_targets_unchanged(self):
        training_spec = load_training_asset_spec()
        policy_spec = load_official_policy_spec()
        targets = make_joint_position_targets(training_spec, policy_spec, [1.0] * 10)

        self.assertAlmostEqual(targets["proximal_pitch_L_Joint"], 0.25)
        self.assertAlmostEqual(targets["proximal_yaw_L_Joint"], -3.1415 + 0.25)
        self.assertEqual(targets["arm1_Joint"], 0.0)
        self.assertEqual(targets["gripper1_Joint"], 0.05)
        self.assertEqual(targets["gripper2_Joint"], -0.05)
    def test_training_urdf_preserves_movable_joint_order_and_meshes(self):
        contract = load_json(CONTRACT_PATH)
        root = ET.parse(TRAINING_URDF_PATH).getroot()
        movable_joints = [joint.attrib["name"] for joint in root.findall("joint") if joint.attrib["type"] != "fixed"]

        self.assertEqual(root.attrib["name"], "tron2a_10dof_hold_arms")
        self.assertEqual(movable_joints, contract["all_joints_18"])
        self.assertTrue(all(path.exists() for path in referenced_mesh_paths(TRAINING_URDF_PATH)))

    def test_asset_spec_exposes_ten_policy_actuators_and_eight_locked_actuators(self):
        contract = load_json(CONTRACT_PATH)
        spec = load_training_asset_spec()

        self.assertEqual(spec.root_link, "base_Link")
        self.assertEqual(spec.initial_root_position_m, 0.85)
        self.assertEqual(tuple(spec.policy_actuators), tuple(contract["policy_joints_10"]))
        self.assertEqual(tuple(spec.locked_actuators), tuple(contract["locked_joints_8"]))
        self.assertTrue(set(spec.policy_actuators).isdisjoint(spec.locked_actuators))
        self.assertEqual(spec.policy_actuators["knee_L_Joint"].effort_limit, 150.0)
        self.assertGreater(spec.policy_actuators["knee_L_Joint"].armature, 0.0)
        self.assertEqual(spec.locked_actuators["gripper1_Joint"].target_position, 0.05)

    def test_offline_inspection_reports_expected_import_contract(self):
        report = inspect_offline_training_asset()

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["root_link"], "base_Link")
        self.assertEqual(report["movable_joint_count"], 18)
        self.assertEqual(report["policy_joint_count"], 10)
        self.assertEqual(report["locked_joint_count"], 8)

    def test_offline_inspection_cli_runs_from_repository_root(self):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, "tools/inspect_imported_robot.py"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"status": "passed"', result.stdout)

    def test_runtime_validation_contract_allows_uncontrolled_pd_to_fall(self):
        contract = runtime_validation_contract()

        self.assertEqual(contract["duration_s"], 10.0)
        self.assertEqual(contract["physics_dt_s"], 0.005)
        self.assertEqual(contract["maximum_locked_joint_error"], 0.2)
        self.assertNotIn("minimum_base_height_m", contract)


if __name__ == "__main__":
    unittest.main()
