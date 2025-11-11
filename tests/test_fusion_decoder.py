import torch

from captionqa.captioning.fusion import FusionHead, FusionConfig
from captionqa.captioning.decoders import CaptionDecoder, CaptionDecoderConfig


def test_fusion_shapes_and_device():
    cfg = FusionConfig(hidden_size=128, dropout=0.0, device="cpu")
    fuse = FusionHead(cfg)
    v = torch.randn(5, 32)
    a = torch.randn(7, 24)
    out = fuse.fuse(v, a)
    assert out.shape == (1, 128)
    assert out.device.type == "cpu"


def test_decoder_accepts_conditioning_without_transformers(monkeypatch):
    # Force decoder into fallback mode by clearing tokenizer/model
    dec = CaptionDecoder(CaptionDecoderConfig())
    dec.model = None
    dec.tokenizer = None
    cond = torch.randn(1, 64)
    text = dec.generate("hello", conditioning=cond)
    assert "hello" in text
    assert "requires transformers" in text

