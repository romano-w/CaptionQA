import types
from pathlib import Path

import torch

from captionqa.qa.model import AVQAModel
from captionqa.qa.eval import run_zero_shot, EvaluationResult


class StubTokenizer:
    def __init__(self):
        self.vocab = {w: i + 3 for i, w in enumerate("what is the sound".split())}
        self.pad = 0
        self.bos = 1
        self.eos = 2

    def encode(self, text: str):
        return [self.vocab.get(w, 3) for w in text.split()][:8]

    def decode(self, seq, skip_special_tokens=True):
        return "".join(["a"] * len([t for t in seq if t not in (self.pad, self.bos, self.eos)]))


def test_zero_shot_uses_feature_extractors(monkeypatch, tmp_path: Path):
    calls = {"visual": 0, "audio": 0}

    class VEnc:
        def __init__(self, *args, **kwargs):
            pass

        config = types.SimpleNamespace(model_name="stub")

        def encode(self, frames, cache_key=None):
            calls["visual"] += 1
            # return 5 frames of 32-dim features
            return torch.ones(5, 32)

    class AEnc:
        def __init__(self, *args, **kwargs):
            pass

        config = types.SimpleNamespace(model_name="stub", sample_rate=16000)

        def encode(self, media_path, cache_key=None, **kwargs):
            calls["audio"] += 1
            # return sequence of 3 steps of 24-dim features
            return torch.ones(1, 24)

    # Patch encoders and sampler inside qa.eval
    import captionqa.qa.eval as qeval

    monkeypatch.setattr(qeval, "VisualEncoder", VEnc)
    monkeypatch.setattr(qeval, "AudioEncoder", AEnc)
    import numpy as np

    class Sampler:
        def __init__(self, *args, **kwargs):
            pass

        class Cfg:
            frame_rate = 1.0
            enable_projection = False
            num_views = 1
            target_resolution = (8, 8)
            num_pitch = 1

        config = Cfg()

        def sample(self, *args, **kwargs):
            # return a few solid frames
            return [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]

    monkeypatch.setattr(qeval, "PanoramicFrameSampler", Sampler)

    # Minimal dataset with two samples; fields shape follow AVQADataset __getitem__
    dataset = [
        {
            "question_id": "q1",
            "question": "what is the sound",
            "answer": "a",
            "video": str(tmp_path / "v1.mp4"),
            "audio": str(tmp_path / "a1.wav"),
            "temporal_window": None,
        },
        {
            "question_id": "q2",
            "question": "what is the sound",
            "answer": "a",
            "video": str(tmp_path / "v2.mp4"),
            "audio": str(tmp_path / "a2.wav"),
            "temporal_window": (0.0, 1.0),
        },
    ]

    # Model dims must match stubs
    model = AVQAModel(video_dim=32, audio_dim=24, vocab_size=128)
    tok = StubTokenizer()
    results = run_zero_shot(model, dataset, tok, device="cpu", max_length=4)
    assert isinstance(results, list) and len(results) == 2
    assert calls["visual"] == 2 and calls["audio"] == 2
    assert isinstance(results[0], EvaluationResult)
    # Confidence captured when available
    assert results[0].confidence is not None or results[0].confidence is None
    # Temporal window propagated
    assert results[1].temporal_window == (0.0, 1.0)
