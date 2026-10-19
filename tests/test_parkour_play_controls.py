"""Tests for keyboard velocity controls in the parkour player."""

from pathlib import Path
import sys

import torch


SCRIPTS_DIR = Path(__file__).parents[1] / "source/instinctlab/instinctlab/tasks/parkour/scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def test_apply_keyboard_velocity_delta_clamps_to_policy_limits():
    from play_controls import apply_keyboard_velocity_delta

    command = torch.tensor([[0.9, 0.0, 0.8]])

    updated = apply_keyboard_velocity_delta(command, linear_delta=0.5, yaw_delta=0.5)

    assert torch.equal(updated, torch.tensor([[1.0, 0.0, 1.0]]))


def test_apply_keyboard_velocity_delta_allows_stopping_and_opposite_turning():
    from play_controls import apply_keyboard_velocity_delta

    command = torch.tensor([[0.2, 0.0, 0.2]])

    updated = apply_keyboard_velocity_delta(command, linear_delta=-0.5, yaw_delta=-0.5)

    assert torch.equal(updated, torch.tensor([[0.0, 0.0, -0.3]]))


def test_play_routes_wasd_events_through_the_clamped_velocity_helper():
    play_source = (SCRIPTS_DIR / "play.py").read_text()

    assert "from play_controls import apply_keyboard_velocity_delta" in play_source
    assert "KeyboardInput.A" in play_source
    assert "KeyboardInput.D" in play_source
    assert "KeyboardInput.F" not in play_source
    assert "KeyboardInput.G" not in play_source
