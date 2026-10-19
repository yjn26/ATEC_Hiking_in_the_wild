"""Tests for direct parkour ONNX replay helpers."""

from pathlib import Path
import sys

import pytest


SCRIPTS_DIR = Path(__file__).parents[1] / "source/instinctlab/instinctlab/tasks/parkour/scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def test_validate_parkour_onnx_model_dir_requires_both_exported_models(tmp_path: Path):
    from onnxer import validate_parkour_onnx_model_dir

    with pytest.raises(FileNotFoundError, match="0-depth_encoder.onnx"):
        validate_parkour_onnx_model_dir(tmp_path)

    (tmp_path / "0-depth_encoder.onnx").touch()
    with pytest.raises(FileNotFoundError, match="actor.onnx"):
        validate_parkour_onnx_model_dir(tmp_path)

    (tmp_path / "actor.onnx").touch()
    assert validate_parkour_onnx_model_dir(tmp_path) == tmp_path


def test_play_uses_the_supported_yaml_environment_loader():
    play_source = (SCRIPTS_DIR / "play.py").read_text()

    assert "from isaaclab.utils.io import load_yaml" in play_source
    assert "load_pickle" not in play_source


def test_play_makes_its_sibling_modules_importable_when_run_via_runpy():
    play_source = (SCRIPTS_DIR / "play.py").read_text()

    assert 'sys.path.append(os.path.dirname(os.path.abspath(__file__)))' in play_source
