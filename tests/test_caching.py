import os
from pathlib import Path

import numpy as np
import torch

from captionqa.captioning.encoders import (
    VisualEncoder,
    VisualEncoderConfig,
    AudioEncoder,
    AudioEncoderConfig,
)
from captionqa.captioning.pipeline import CaptioningPipeline


def test_visual_encoder_caching_roundtrip(tmp_path: Path):
    cfg = VisualEncoderConfig(
        model_name="__disabled__",
        cache_dir=str(tmp_path / "vis"),
        use_cache=True,
        batch_size=4,
    )
    enc = VisualEncoder(cfg)

    # Two tiny RGB frames
    frames = [
        (np.zeros((8, 8, 3), dtype=np.uint8)),
        (np.ones((8, 8, 3), dtype=np.uint8) * 255),
    ]
    key = "unit_test_key_vis"

    out1 = enc.encode(frames, cache_key=key)
    # Ensure cache file exists
    cache_file = Path(cfg.cache_dir) / f"{key}.pt"
    assert cache_file.exists()

    # Delete encoder to ensure we must load from cache
    del enc
    enc2 = VisualEncoder(cfg)
    out2 = enc2.encode(frames, cache_key=key)

    assert torch.allclose(out1.cpu(), out2.cpu())


def test_audio_encoder_prefers_cache(tmp_path: Path):
    cfg = AudioEncoderConfig(
        model_name="__disabled__",
        cache_dir=str(tmp_path / "aud"),
        use_cache=True,
    )
    enc = AudioEncoder(cfg)

    key = "unit_test_key_aud"
    expected = torch.randn(3, 5)
    cache_file = Path(cfg.cache_dir) / f"{key}.pt"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(expected, cache_file)

    # Provide a non-existent path to force loader fallback; cache should be used instead.
    out = enc.encode(str(tmp_path / "no_such_video.mp4"), cache_key=key)
    assert torch.allclose(out.cpu(), expected)


def test_pipeline_passes_cache_keys(tmp_path: Path):
    # Dummy sampler that returns one frame regardless of input
    class DummySampler:
        def __init__(self):
            class C:
                frame_rate = 1.0
                enable_projection = False
                num_views = 1
                target_resolution = (8, 8)

            self.config = C()

        def sample(self, _: str):
            return [np.zeros((8, 8, 3), dtype=np.uint8)]

    vis_calls = {}
    aud_calls = {}

    vcfg = VisualEncoderConfig(model_name="__disabled__", cache_dir=str(tmp_path / "vis"))
    acfg = AudioEncoderConfig(model_name="__disabled__", cache_dir=str(tmp_path / "aud"))
    v = VisualEncoder(vcfg)
    a = AudioEncoder(acfg)

    def v_encode(frames, *, cache_key=None):
        vis_calls["cache_key"] = cache_key
        return torch.ones(len(frames), 1)

    def a_encode(path, *, cache_key=None):
        aud_calls["cache_key"] = cache_key
        return torch.ones(1, 1)

    v.encode = v_encode  # type: ignore
    a.encode = a_encode  # type: ignore

    class DummyDecoder:
        class Cfg:
            max_new_tokens = 8

        def __init__(self):
            self.config = DummyDecoder.Cfg()

        def generate(self, prompt: str) -> str:
            return prompt

    p = CaptioningPipeline(sampler=DummySampler(), visual_encoder=v, audio_encoder=a, decoder=DummyDecoder())
    # Provide a path that exists check is not enforced here (pipeline doesn't check)
    _ = p.generate(str(tmp_path / "video.mp4"))

    assert isinstance(vis_calls.get("cache_key"), str)
    assert isinstance(aud_calls.get("cache_key"), str)
    assert len(vis_calls["cache_key"]) == 40
    assert len(aud_calls["cache_key"]) == 40


def test_cli_applies_cache_overrides(tmp_path: Path, monkeypatch):
    from captionqa.captioning import cli

    captured = {}

    def fake_generate(video, *, config, prompt=None, max_new_tokens=None):  # noqa: D401
        captured["config"] = config
        return "ok"

    monkeypatch.setattr(cli, "generate_captions", fake_generate)

    # Create a dummy file for the existence check
    dummy = tmp_path / "vid.mp4"
    dummy.write_bytes(b"")

    cache_root = tmp_path / "cache"
    argv = [str(dummy), "--no-cache", "--cache-dir", str(cache_root)]
    cli.main(argv)

    cfg = captured["config"]
    assert cfg.visual_encoder.use_cache is False
    assert cfg.audio_encoder.use_cache is False
    assert str(cfg.visual_encoder.cache_dir).endswith("visual")
    assert str(cfg.audio_encoder.cache_dir).endswith("audio")

