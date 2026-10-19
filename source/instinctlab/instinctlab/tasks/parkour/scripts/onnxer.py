from __future__ import annotations

import numpy as np
import os
import torch
from typing import TYPE_CHECKING
from pathlib import Path

import onnxruntime as ort

if TYPE_CHECKING:
    from typing import Callable


def validate_parkour_onnx_model_dir(model_path: str | Path) -> Path:
    """Return the directory containing the exported parkour ONNX models."""
    model_dir = Path(model_path).expanduser().resolve()
    if model_dir.is_file():
        if model_dir.name != "actor.onnx":
            raise ValueError("--onnx_model_dir may only reference actor.onnx or its containing directory.")
        model_dir = model_dir.parent

    for filename in ("0-depth_encoder.onnx", "actor.onnx"):
        if not (model_dir / filename).is_file():
            raise FileNotFoundError(f"Missing required parkour ONNX model: {model_dir / filename}")
    return model_dir


def load_parkour_onnx_model(
    model_dir: str, get_subobs_func: Callable, depth_shape: tuple, proprio_slice: slice
) -> Callable:
    """Load the ONNX model as policy, but only for parkour task setting."""
    ort_providers = ort.get_available_providers()
    encoder = ort.InferenceSession(os.path.join(model_dir, "0-depth_encoder.onnx"), providers=ort_providers)
    actor = ort.InferenceSession(os.path.join(model_dir, "actor.onnx"), providers=ort_providers)
    actor_input_name = actor.get_inputs()[0].name
    logged_input_shape = False

    def policy(obs: torch.Tensor) -> torch.Tensor:
        nonlocal logged_input_shape
        depth_image_input = get_subobs_func(obs)
        depth_image_input = depth_image_input.cpu().numpy()
        depth_image_input = depth_image_input.reshape((-1, *depth_shape))
        depth_image_output = encoder.run(None, {encoder.get_inputs()[0].name: depth_image_input})[0]
        actor_input = np.concatenate(
            [
                obs.cpu().numpy()[:, proprio_slice],
                depth_image_output,
            ],
            axis=1,
        )
        if not logged_input_shape:
            print(f"[INFO]: Parkour ONNX actor input shape: {actor_input.shape}")
            logged_input_shape = True
        actor_output = actor.run(None, {actor_input_name: actor_input})[0]
        return torch.from_numpy(actor_output).to(obs.device)

    return policy
