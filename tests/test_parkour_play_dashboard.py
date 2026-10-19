"""Tests for the parkour replay Qt dashboard helpers."""

from pathlib import Path
import sys
import types

import numpy as np


SCRIPTS_DIR = Path(__file__).parents[1] / "source/instinctlab/instinctlab/tasks/parkour/scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def test_depth_to_rgba_normalizes_depth_and_marks_invalid_pixels():
    from play_dashboard import depth_to_rgba

    rgba = depth_to_rgba(np.array([[0.0, 2.5], [5.0, np.nan]], dtype=np.float32), max_depth_m=5.0)

    assert rgba.shape == (2, 2, 4)
    assert np.array_equal(rgba[0, 0], np.array([0, 0, 0, 255], dtype=np.uint8))
    assert np.array_equal(rgba[1, 0], np.array([255, 255, 255, 255], dtype=np.uint8))
    assert np.array_equal(rgba[1, 1], np.array([0, 0, 0, 255], dtype=np.uint8))


def test_depth_to_rgba_defaults_to_the_normalized_policy_depth_range():
    from play_dashboard import depth_to_rgba

    rgba = depth_to_rgba(np.array([[0.0, 0.5, 1.0]], dtype=np.float32))

    assert np.array_equal(rgba[0, 0], np.array([0, 0, 0, 255], dtype=np.uint8))
    assert np.array_equal(rgba[0, 1], np.array([127, 127, 127, 255], dtype=np.uint8))
    assert np.array_equal(rgba[0, 2], np.array([255, 255, 255, 255], dtype=np.uint8))


def test_format_velocity_reports_linear_and_yaw_components():
    from play_dashboard import format_velocity

    assert format_velocity("Command", np.array([0.5, -0.1, 0.25])) == "Command: vx +0.50 m/s | vy -0.10 m/s | wz +0.25 rad/s"


def test_play_enables_the_depth_dashboard_by_default_for_gui_runs():
    play_source = (SCRIPTS_DIR / "play.py").read_text()

    assert "--no_depth_viewer" in play_source
    assert "from play_dashboard import QtDepthDashboard" in play_source
    assert "dashboard.update(" in play_source


def test_dashboard_submits_rgba_bytes_to_kit_image_provider(monkeypatch):
    from play_dashboard import QtDepthDashboard

    class FakeProvider:
        def __init__(self):
            self.data = None

        def set_bytes_data(self, data, size):
            self.data = (data, size)

    class FakeWindow:
        def __init__(self, *args, **kwargs):
            self.frame = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeStack:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeLabel:
        def __init__(self, text):
            self.text = text

    fake_ui = types.SimpleNamespace(
        ByteImageProvider=FakeProvider,
        Window=FakeWindow,
        VStack=FakeStack,
        ImageWithProvider=lambda *args, **kwargs: object(),
        Label=FakeLabel,
        IwpFillPolicy=types.SimpleNamespace(IWP_STRETCH=object()),
        Pixel=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "omni", types.SimpleNamespace(ui=fake_ui))
    monkeypatch.setitem(sys.modules, "omni.ui", fake_ui)

    dashboard = QtDepthDashboard()
    dashboard.update(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )

    pixels, size = dashboard._image_provider.data
    assert size == [2, 2]
    assert len(pixels) == 2 * 2 * 4
