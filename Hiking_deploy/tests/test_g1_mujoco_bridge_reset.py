from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


BRIDGE_PATH = Path(__file__).parents[1] / "sim2sim" / "g1_mujoco_bridge.py"
SPEC = importlib.util.spec_from_file_location("g1_mujoco_bridge", BRIDGE_PATH)
bridge_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bridge_module)


def make_bridge(*, upright: bool, tilt_deg: float = 0.0) -> object:
    bridge = bridge_module.G1MujocoBridge.__new__(bridge_module.G1MujocoBridge)
    xmat = np.eye(3).reshape(-1)
    if upright and tilt_deg:
        angle = np.deg2rad(tilt_deg)
        xmat = np.array(
            [np.cos(angle), 0.0, np.sin(angle), 0.0, 1.0, 0.0, -np.sin(angle), 0.0, np.cos(angle)]
        )
    elif not upright:
        xmat = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0])

    bridge.args = SimpleNamespace(spawn_x=1.0, spawn_y=-2.0, spawn_height=0.76)
    bridge.spawn_pos = np.array([1.0, -2.0, 0.76])
    bridge.data = SimpleNamespace(
        qpos=np.array([3.0, 4.0, 0.76, 0.5, 0.5, 0.5, 0.5] + [0.0] * 29),
        qvel=np.ones(35),
        xfrc_applied=np.ones((2, 6)),
        xmat=np.stack((xmat, np.eye(3).reshape(-1))),
    )
    bridge.model = object()
    bridge.pelvis_bid = 0
    bridge.torso_bid = 1
    bridge.base_height = 0.76
    bridge.hold_q = np.linspace(-0.3, 0.3, 29)
    bridge.gantry_xy = np.zeros(2)
    bridge.gantry_on = False
    bridge.joy_ly = bridge.joy_lx = bridge.joy_rx = 1.0
    bridge._turn_until = 1.0
    bridge.btn_until = {1: 1.0}
    bridge._btn_pulse = 1.2
    bridge._recovery_until = 0.0
    bridge._reset_recovery_stage = "idle"
    bridge._reset_recovery_l1_at = 0.0
    bridge._reset_recovery_release_at = 0.0
    return bridge


def test_reset_to_spawn_reanchors_an_upright_stationary_robot(monkeypatch):
    bridge = make_bridge(upright=True)
    bridge.data.qvel[:] = 0.0
    monkeypatch.setattr(bridge_module.mujoco, "mj_forward", lambda model, data: None)

    assert bridge.reset_to_spawn_if_standing() is True
    assert np.allclose(bridge.data.qpos[:7], [1.0, -2.0, 0.76, 1.0, 0.0, 0.0, 0.0])
    assert np.allclose(bridge.data.qvel, 0.0)
    assert np.allclose(bridge.data.qpos[7:], bridge.hold_q)
    assert np.allclose(bridge.data.xfrc_applied, 0.0)
    assert bridge.gantry_on is True
    assert np.allclose(bridge.gantry_xy, [1.0, -2.0])
    assert (bridge.joy_ly, bridge.joy_lx, bridge.joy_rx) == (0.0, 0.0, 0.0)


def test_reset_to_spawn_forces_a_fallen_moving_robot_into_recovery_stand(monkeypatch):
    bridge = make_bridge(upright=False)
    monkeypatch.setattr(bridge_module.mujoco, "mj_forward", lambda model, data: None)

    assert bridge.reset_to_spawn_if_standing() is True
    assert np.allclose(bridge.data.qpos[:7], [1.0, -2.0, 0.76, 1.0, 0.0, 0.0, 0.0])
    assert np.allclose(bridge.data.qvel, 0.0)
    assert bridge.gantry_on is True
    assert bridge.btn_until[bridge_module.BTN_R1] > 0.0


def test_r_key_ignores_key_repeat_during_the_reset_debounce(monkeypatch):
    bridge = make_bridge(upright=True)
    bridge._last_reset_at = float("-inf")
    bridge._reset_debounce = 0.5
    calls = []
    monkeypatch.setattr(bridge_module.time, "time", lambda: 100.0)
    monkeypatch.setattr(bridge, "reset_to_spawn_if_standing", lambda: calls.append("reset"))

    bridge.key_callback(ord("r"))
    bridge.key_callback(ord("r"))

    assert calls == ["reset"]


def test_reset_to_spawn_accepts_a_slightly_tilted_standing_robot(monkeypatch):
    bridge = make_bridge(upright=True, tilt_deg=12.1)
    bridge.data.qvel[:] = 0.0
    monkeypatch.setattr(bridge_module.mujoco, "mj_forward", lambda model, data: None)

    assert bridge.reset_to_spawn_if_standing() is True


def test_reset_recovery_returns_to_free_parkour_after_stand_and_l1(monkeypatch):
    bridge = make_bridge(upright=False)
    monkeypatch.setattr(bridge_module.time, "time", lambda: 100.0)
    monkeypatch.setattr(bridge_module.mujoco, "mj_forward", lambda model, data: None)

    bridge.reset_to_spawn_if_standing()
    bridge._advance_reset_recovery(101.9)
    assert bridge.gantry_on is True
    assert bridge_module.BTN_L1 not in bridge.btn_until

    bridge._advance_reset_recovery(102.0)
    assert bridge.btn_until[bridge_module.BTN_L1] == 103.2
    assert bridge.gantry_on is True

    bridge.data.xfrc_applied[:] = 1.0
    bridge._advance_reset_recovery(104.0)
    assert bridge.gantry_on is False
    assert bridge._reset_recovery_stage == "idle"
    assert np.allclose(bridge.data.xfrc_applied, 0.0)
