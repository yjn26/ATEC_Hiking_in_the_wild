import unittest

from tools.validate_asset_contract import validate_contract


class AssetContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = validate_contract()

    def test_full_contract_passes(self):
        self.assertEqual(self.report["status"], "passed")
        self.assertEqual(self.report["errors"], [])

    def test_joint_partition_is_complete(self):
        self.assertEqual(self.report["movable_joint_count"], 18)
        self.assertEqual(self.report["policy_joint_count"], 10)
        self.assertEqual(self.report["locked_joint_count"], 8)
        self.assertEqual(
            set(self.report["policy_joint_order"]) | set(self.report["locked_joint_order"]),
            set(self.report["all_joint_order"]),
        )

    def test_locked_targets_are_pending_hardware_approval(self):
        self.assertEqual(self.report["locked_pose_approval_status"], "pending_mechanical_safety_review")
        self.assertEqual(self.report["locked_joint_targets"]["gripper1_Joint"], 0.05)
        self.assertEqual(self.report["locked_joint_targets"]["gripper2_Joint"], -0.05)


if __name__ == "__main__":
    unittest.main()
