import numpy as np
from pathlib import Path

from captionqa.captioning.panorama import PanoramicFrameSampler, PanoramaSamplingConfig


def _write_video(path: Path, frames: int = 3, size=(64, 128), fps: int = 8):
    try:
        import cv2  # type: ignore
    except Exception:  # pragma: no cover
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (size[1], size[0]))
    if not writer.isOpened():  # pragma: no cover
        return None
    for i in range(frames):
        v = int(50 + i * 50)
        frame = np.full((size[0], size[1], 3), v, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_sampling_counts_with_pitch(tmp_path: Path):
    video = tmp_path / "vid.avi"
    if _write_video(video) is None:
        import pytest

        pytest.skip("OpenCV VideoWriter not available")

    # Ensure we sample every frame
    cfg = PanoramaSamplingConfig(
        frame_rate=8.0,
        target_resolution=(32, 64),
        enable_projection=True,
        fov_degrees=90.0,
        num_views=5,
        num_pitch=2,
        pitch_min_degrees=-20.0,
        pitch_max_degrees=20.0,
        roll_degrees=0.0,
    )
    sampler = PanoramicFrameSampler(cfg)
    out = sampler.sample(str(video))
    # 3 frames * 5 yaw * 2 pitch
    assert len(out) == 3 * 5 * 2
    assert out[0].shape == (32, 64, 3)

