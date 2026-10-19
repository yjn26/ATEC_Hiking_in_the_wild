"""Validated interface to the published SF_TRON2A ONNX policy contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    from assets.tron2a_10dof import TrainingAssetSpec


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POLICY_MANIFEST_PATH = REPOSITORY_ROOT / "assets/resources/tron2a_10dof/SF_TRON2A_POLICY_MANIFEST.json"
DEFAULT_POLICY_ROOT = REPOSITORY_ROOT / "third_party/tron2_rl_deploy_python"


@dataclass(frozen=True)
class OfficialPolicySpec:
    source_commit: str
    action_joint_names: tuple[str, ...]
    default_joint_positions: dict[str, float]
    action_scale_position: float
    loop_frequency_hz: int
    decimation: int
    encoder_input_size: int
    encoder_output_size: int
    policy_input_size: int
    policy_output_size: int


class OfficialOnnxPolicy:
    """Stateful inference wrapper matching the upstream SF_TRON2A controller."""

    def __init__(self, policy_root: Path):
        import numpy as np
        import onnxruntime as ort

        self.spec = load_official_policy_spec()
        encoder_path, policy_path = verify_policy_checkout(policy_root)
        self._np = np
        self._encoder = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
        self._policy = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        self._history: np.ndarray | None = None
        self._last_actions = np.zeros(self.spec.policy_output_size, dtype=np.float32)

    def infer(self, observation: Sequence[float], commands: Sequence[float] = (0.0, 0.0, 0.0)) -> list[float]:
        obs = self._np.asarray(observation, dtype=self._np.float32)
        commands_array = self._np.asarray(commands, dtype=self._np.float32)
        if obs.shape != (42,):
            raise ValueError(f"official policy requires a 42-element observation, received {obs.shape}")
        if commands_array.shape != (3,):
            raise ValueError(f"official policy requires three command values, received {commands_array.shape}")
        if self._history is None:
            self._history = self._np.tile(obs, self.spec.encoder_input_size // obs.size)
        else:
            self._history = self._np.concatenate((self._history[obs.size :], obs))
        encoder_output = self._encoder.run(None, {self._encoder.get_inputs()[0].name: self._history})[0].reshape(-1)
        policy_input = self._np.concatenate((encoder_output, obs, commands_array)).astype(self._np.float32)
        actions = self._policy.run(None, {self._policy.get_inputs()[0].name: policy_input})[0].reshape(-1)
        if actions.shape != (self.spec.policy_output_size,) or not self._np.isfinite(actions).all():
            raise RuntimeError(f"official policy produced invalid action output: {actions}")
        self._last_actions = actions.astype(self._np.float32)
        return self._last_actions.tolist()


def _manifest() -> dict:
    return json.loads(POLICY_MANIFEST_PATH.read_text(encoding="utf-8"))


def load_official_policy_spec() -> OfficialPolicySpec:
    manifest = _manifest()
    control = manifest["control"]
    network = manifest["network"]
    return OfficialPolicySpec(
        source_commit=manifest["source"]["commit"],
        action_joint_names=tuple(manifest["action_joint_names"]),
        default_joint_positions=dict(manifest["default_joint_positions"]),
        action_scale_position=float(control["action_scale_position"]),
        loop_frequency_hz=int(control["loop_frequency_hz"]),
        decimation=int(control["decimation"]),
        encoder_input_size=int(network["encoder_input_size"]),
        encoder_output_size=int(network["encoder_output_size"]),
        policy_input_size=int(network["policy_input_size"]),
        policy_output_size=int(network["policy_output_size"]),
    )


def runtime_joint_ids(runtime_joint_names: Sequence[str], joint_names: Iterable[str]) -> list[int]:
    names = list(runtime_joint_names)
    if len(names) != len(set(names)):
        raise ValueError("Isaac runtime joint names must be unique")
    missing = [name for name in joint_names if name not in names]
    if missing:
        raise ValueError(f"Isaac runtime is missing required joints: {missing}")
    return [names.index(name) for name in joint_names]


def make_joint_position_targets(
    training_spec: TrainingAssetSpec, policy_spec: OfficialPolicySpec, actions: Sequence[float]
) -> dict[str, float]:
    if tuple(training_spec.policy_actuators) != policy_spec.action_joint_names:
        raise ValueError("official policy joint order differs from the phase-1 policy contract")
    if len(actions) != policy_spec.policy_output_size:
        raise ValueError(f"expected {policy_spec.policy_output_size} policy actions, received {len(actions)}")
    targets = {
        name: policy_spec.default_joint_positions[name] + policy_spec.action_scale_position * float(action)
        for name, action in zip(policy_spec.action_joint_names, actions, strict=True)
    }
    targets.update(
        {
            name: actuator.target_position
            for name, actuator in training_spec.locked_actuators.items()
            if actuator.target_position is not None
        }
    )
    return targets


def policy_pd_gains(joint_name: str) -> tuple[float, float]:
    control = _manifest()["control"]
    if "yaw" in joint_name or "ankle" in joint_name:
        return float(control["light_stiffness"]), float(control["light_damping"])
    return float(control["leg_stiffness"]), float(control["leg_damping"])


def verify_policy_checkout(policy_root: Path) -> tuple[Path, Path]:
    manifest = _manifest()
    model_dir = policy_root / manifest["source"]["model_directory"]
    expected_files = manifest["files"]
    for name, expected_sha256 in expected_files.items():
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"official policy input missing: {path}")
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(f"official policy input hash mismatch for {path}: {actual_sha256}")
    return model_dir / "encoder.onnx", model_dir / "policy.onnx"
