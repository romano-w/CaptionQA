import json
from pathlib import Path

import numpy as np


def _write_dummy_video(path: Path, *, width: int = 128, height: int = 64, frames: int = 8, fps: int = 8):
    try:
        import cv2  # type: ignore
    except Exception:  # pragma: no cover - skip if OpenCV missing
        return None

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():  # pragma: no cover
        return None
    # Write solid-color frames with slight gradient so mean != 0
    for i in range(frames):
        val = int(10 + i * (200 / max(frames - 1, 1)))
        frame = np.full((height, width, 3), val, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_cli_end_to_end_with_disabled_models(tmp_path: Path, monkeypatch):
    # Create a dummy video at a path containing spaces to exercise Windows path handling
    video_dir = tmp_path / "with space"
    video_path = video_dir / "dummy.avi"
    if _write_dummy_video(video_path) is None:
        import pytest

        pytest.skip("OpenCV VideoWriter not available")

    # Minimal config disabling remote models to exercise deterministic fallback
    config = {
        "visual_encoder": {"model_name": "__disabled__"},
        "audio_encoder": {"model_name": "__disabled__", "sample_rate": 16000},
        "decoder": {"model_name": "__disabled__", "max_new_tokens": 8},
        # keep panorama defaults (projection enabled) to validate multiple views
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")

    from captionqa.captioning import cli

    caption = cli.main([str(video_path), "--config", str(cfg_path)])
    # Check deterministic fallback marker and that visual stats are present
    assert "Caption generation requires transformers models." in caption
    assert "Visual embedding stats:" in caption


def test_eval_cli_on_jsonl(tmp_path: Path):
    preds = tmp_path / "preds.jsonl"
    refs = tmp_path / "refs.jsonl"
    preds.write_text("{" + '"id":"1","prediction":"a black scene"' + "}\n", encoding="utf-8")
    refs.write_text("{" + '"id":"1","references":["a black 360-degree scene"]' + "}\n", encoding="utf-8")

    from captionqa.evaluation import run

    summary = run.main([
        "--task",
        "captioning",
        "--preds",
        str(preds),
        "--refs",
        str(refs),
    ])
    assert summary["task"] == "captioning"
    assert "metrics" in summary and "bleu" in summary["metrics"]
