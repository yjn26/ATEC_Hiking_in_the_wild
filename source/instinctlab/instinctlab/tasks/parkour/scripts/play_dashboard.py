"""Qt dashboard for inspecting parkour policy replay state."""

import numpy as np


def depth_to_rgba(depth_m: np.ndarray, max_depth_m: float = 1.0) -> np.ndarray:
    """Convert normalized policy depth to an opaque grayscale RGBA image."""
    depth_m = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth_m)
    normalized = np.clip(np.nan_to_num(depth_m, nan=0.0) / max_depth_m, 0.0, 1.0)
    grayscale = (normalized * 255.0).astype(np.uint8)
    grayscale[~valid] = 0
    return np.stack((grayscale, grayscale, grayscale, np.full_like(grayscale, 255)), axis=-1)


def format_velocity(label: str, velocity: np.ndarray) -> str:
    """Format base-frame linear and yaw velocities for the dashboard."""
    vx, vy, wz = np.asarray(velocity, dtype=np.float32)
    return f"{label}: vx {vx:+.2f} m/s | vy {vy:+.2f} m/s | wz {wz:+.2f} rad/s"


class QtDepthDashboard:
    """Non-modal Kit Qt window that displays depth and velocity state during replay."""

    def __init__(self, title: str = "Parkour Replay Monitor"):
        import omni.ui as ui

        self._ui = ui
        self._image_provider = ui.ByteImageProvider()
        self._window = ui.Window(title, width=480, height=390)
        with self._window.frame:
            with ui.VStack(spacing=8):
                self._depth_image = ui.ImageWithProvider(
                    self._image_provider,
                    fill_policy=ui.IwpFillPolicy.IWP_STRETCH,
                    height=ui.Pixel(270),
                )
                self._command_label = ui.Label("")
                self._actual_label = ui.Label("")

    def update(self, depth_m: np.ndarray, command: np.ndarray, actual: np.ndarray) -> None:
        """Refresh the image and velocity labels."""
        rgba = np.ascontiguousarray(depth_to_rgba(depth_m))
        height, width = rgba.shape[:2]
        self._image_provider.set_bytes_data(rgba.reshape(-1).tolist(), [width, height])
        self._command_label.text = format_velocity("Command", command)
        self._actual_label.text = format_velocity("Actual", actual)
