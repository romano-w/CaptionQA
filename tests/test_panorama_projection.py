import numpy as np

from captionqa.captioning.panorama import EquirectangularProjector


def _make_gradient(width: int, height: int):
    # Horizontal gradient from 0..255, constant over rows
    x = np.linspace(0, 255, width, dtype=np.float32)
    row = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=-1)
    img = np.repeat(row[None, :, :], height, axis=0)
    return img.astype(np.uint8)


def test_projection_shape_and_types():
    w, h = 256, 128
    src = _make_gradient(w, h)
    proj = EquirectangularProjector(target_resolution=(64, 128), fov_degrees=90.0)
    out0 = proj.project(src, yaw=0.0)
    out90 = proj.project(src, yaw=90.0)
    assert out0.shape == (64, 128, 3)
    assert out90.shape == (64, 128, 3)
    assert out0.dtype == src.dtype


def test_center_column_matches_yaw_position():
    w, h = 400, 100
    src = _make_gradient(w, h)
    proj = EquirectangularProjector(target_resolution=(60, 120), fov_degrees=90.0)

    for yaw in (0.0, 90.0, 180.0, 270.0):
        out = proj.project(src, yaw=yaw)
        center_col = out[:, out.shape[1] // 2, 0].astype(np.float32)
        # In equirectangular, lon=0 maps to center (W/2); yaw rotates lon by yaw.
        expected_u = int((((yaw % 360.0) / 360.0) + 0.5) % 1.0 * w)
        expected_val = src[0, expected_u, 0].astype(np.float32)
        diff = np.abs(center_col.mean() - expected_val)
        assert diff <= 3.0, f"center column deviates too much at yaw={yaw}: {diff}"
