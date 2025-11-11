import types

from captionqa.captioning.config import CaptioningConfig, QwenVLConfig, PanoramaSamplingConfig
from captionqa.captioning.pipeline import generate_captions


def test_qwen_engine_fallback(monkeypatch, tmp_path):
    # Replace engine class to avoid heavy imports and sampling
    import captionqa.captioning.engines.qwen_vl as qwen_mod

    class StubEngine:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def from_configs(cls, **kwargs):
            return cls()

        def generate(self, video_path: str, *, prompt=None, max_new_tokens=None):
            return f"ok: {prompt or 'no-prompt'}"

    monkeypatch.setattr(qwen_mod, "QwenVLEngine", StubEngine)
    # Also patch symbol imported in pipeline
    import captionqa.captioning.pipeline as pipe_mod
    monkeypatch.setattr(pipe_mod, "QwenVLEngine", StubEngine)

    cfg = CaptioningConfig.from_defaults(
        panorama=PanoramaSamplingConfig(enable_projection=False),
    )
    cfg.engine = "qwen_vl"
    cfg.qwen_vl = QwenVLConfig(model_name="stub")
    out = generate_captions(str(tmp_path / "video.mp4"), config=cfg, prompt="hello")
    assert out.startswith("ok: hello")
