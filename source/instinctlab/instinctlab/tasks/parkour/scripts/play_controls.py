"""Keyboard controls shared by the parkour policy player."""

import torch


MAX_FORWARD_SPEED_MPS = 1.0
MAX_YAW_RATE_RADPS = 1.0


def apply_keyboard_velocity_delta(
    command: torch.Tensor, linear_delta: float = 0.0, yaw_delta: float = 0.0
) -> torch.Tensor:
    """Apply and clamp a velocity delta to the parkour policy's training limits."""
    command = command.clone()
    command[:, 0].add_(linear_delta).clamp_(0.0, MAX_FORWARD_SPEED_MPS)
    command[:, 2].add_(yaw_delta).clamp_(-MAX_YAW_RATE_RADPS, MAX_YAW_RATE_RADPS)
    return command
