"""Utilities for sampling and projecting panoramic video frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - OpenCV is optional at runtime
    import cv2
except Exception:  # pragma: no cover - fallback if OpenCV is unavailable
    cv2 = None


@dataclass
class PanoramaSamplingConfig:
    """Configuration describing how panoramic frames should be sampled.

    Attributes
    ----------
    frame_rate:
        Number of frames per second to sample. A value of ``0`` falls back to
        uniform sampling over the entire duration.
    target_resolution:
        Desired ``(height, width)`` of the sampled frames.
    enable_projection:
        Whether to project equirectangular frames into multiple perspective
        views. If ``False`` the original panoramic frame is returned.
    fov_degrees:
        Horizontal field-of-view used when generating perspective crops.
    num_views:
        How many discrete yaw angles to sample around the panorama when
        ``enable_projection`` is ``True``.
    """

    frame_rate: float = 1.0
    target_resolution: Tuple[int, int] = (512, 1024)
    enable_projection: bool = True
    fov_degrees: float = 90.0
    num_views: int = 4


class EquirectangularProjector:
    """Projects equirectangular panoramas into perspective views.

    The implementation is intentionally lightweight—enough to provide a
    repeatable projection interface without requiring heavy GPU kernels. When
    OpenCV is missing we fall back to returning the original frame slice.
    """

    def __init__(self, target_resolution: Tuple[int, int], fov_degrees: float):
        self.target_resolution = target_resolution
        self.fov_radians = np.deg2rad(fov_degrees)

    def project(self, frame: np.ndarray, yaw: float) -> np.ndarray:
        """Project ``frame`` around the provided ``yaw`` angle."""

        if cv2 is None:  # pragma: no cover - exercised when OpenCV missing
            return frame

        height, width = self.target_resolution
        # Compute the x-offset in the equirectangular image corresponding to
        # the requested yaw. The mapping is linear in the panorama.
        yaw = yaw % 360.0
        center_x = int((yaw / 360.0) * frame.shape[1])
        half_width = int((self.fov_radians / (2 * np.pi)) * frame.shape[1])

        width_px = frame.shape[1]
        left = center_x - half_width
        right = center_x + half_width
        if right <= left:
            right = left + 1

        # Handle wrap-around by concatenating slices from both sides.
        if left < 0:
            wrapped = np.concatenate(
                [frame[:, left:], frame[:, : max(right, 0)]], axis=1
            )
        elif right > width_px:
            wrapped = np.concatenate(
                [frame[:, left:], frame[:, : (right - width_px)]], axis=1
            )
        else:
            wrapped = frame[:, left:right]

        resized = cv2.resize(wrapped, (width, height), interpolation=cv2.INTER_AREA)
        return resized


class PanoramicFrameSampler:
    """Sample frames from a 360° equirectangular video."""

    def __init__(self, config: PanoramaSamplingConfig):
        self.config = config
        self.projector = None
        if config.enable_projection:
            self.projector = EquirectangularProjector(
                target_resolution=config.target_resolution,
                fov_degrees=config.fov_degrees,
            )

    def _iter_frames(self, video_path: str) -> Iterable[np.ndarray]:
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is required for frame sampling but is not available. "
                "Install 'opencv-python' to enable panoramic sampling."
            )

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video file: {video_path}")

        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            step = 1
            if self.config.frame_rate and fps:
                step = max(int(fps / self.config.frame_rate), 1)
            elif total_frames > 0:
                step = max(int(total_frames / 32), 1)

            index = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if index % step == 0:
                    yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                index += 1
        finally:
            capture.release()

    def sample(self, video_path: str) -> List[np.ndarray]:
        """Sample and optionally project frames from ``video_path``."""

        frames = list(self._iter_frames(video_path))
        if not frames:
            return []

        if not self.projector:
            return [self._resize(frame) for frame in frames]

        projected: List[np.ndarray] = []
        yaw_angles = np.linspace(0, 360, num=self.config.num_views, endpoint=False)
        for frame in frames:
            for yaw in yaw_angles:
                projected.append(self.projector.project(frame, float(yaw)))
        return [self._resize(frame) for frame in projected]

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if cv2 is None:
            return frame
        height, width = self.config.target_resolution
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def summarise_frames(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Compute a simple statistical descriptor over sampled frames."""

    if not frames:
        return np.zeros((1, 3), dtype=np.float32)

    stacked = np.stack(frames).astype(np.float32) / 255.0
    mean_rgb = stacked.mean(axis=(0, 1, 2))
    std_rgb = stacked.std(axis=(0, 1, 2))
    return np.concatenate([mean_rgb, std_rgb])[None, :]

