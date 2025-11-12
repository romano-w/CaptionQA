def test_temporal_window_passed_to_qwen_engine(monkeypatch, tmp_path):
    from captionqa.captioning.pipeline import generate_captions
    from captionqa.captioning.config import CaptioningConfig, PanoramaSamplingConfig

    # Stub engine to capture start/end
    import captionqa.captioning.engines.qwen_vl as qmod

    captured = {}

    class StubEngine:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def from_configs(cls, *args, **kwargs):
            return cls()

        def generate(self, video_path, *, prompt=None, max_new_tokens=None, start_sec=None, end_sec=None):
            captured["start_sec"] = start_sec
            captured["end_sec"] = end_sec
            return "ok"

    monkeypatch.setattr(qmod, "QwenVLEngine", StubEngine)
    import captionqa.captioning.pipeline as pipe
    monkeypatch.setattr(pipe, "QwenVLEngine", StubEngine)

    cfg = CaptioningConfig.from_defaults(panorama=PanoramaSamplingConfig(enable_projection=False), engine="qwen_vl")
    _ = generate_captions(str(tmp_path / "v.mp4"), config=cfg, temporal_window=(1.0, 2.0))
    assert captured["start_sec"] == 1.0 and captured["end_sec"] == 2.0

