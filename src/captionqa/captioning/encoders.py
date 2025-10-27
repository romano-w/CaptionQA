"""Audio/visual encoder backbones for the captioning pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Sequence

import numpy as np
import torch

try:  # pragma: no cover - torchaudio is optional
    import torchaudio
except Exception:  # pragma: no cover - graceful fallback
    torchaudio = None

try:  # pragma: no cover - transformers are optional during tests
    from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor
except Exception:  # pragma: no cover - allow running without transformers
    AutoFeatureExtractor = None
    AutoProcessor = None
    AutoModel = None

try:  # pragma: no cover - ffmpeg may be missing in lightweight envs
    import ffmpeg
except Exception:  # pragma: no cover - fallback path
    ffmpeg = None


@dataclass
class VisualEncoderConfig:
    """Configuration for the visual backbone."""

    model_name: str = "openai/clip-vit-base-patch32"
    device: Optional[str] = None


@dataclass
class AudioEncoderConfig:
    """Configuration for the audio backbone."""

    model_name: str = "microsoft/wavlm-base-plus"
    sample_rate: int = 16000
    device: Optional[str] = None


class VisualEncoder:
    """Image encoder built on top of Hugging Face ``transformers`` models."""

    def __init__(self, config: VisualEncoderConfig):
        self.config = config
        self.device = torch.device(
            config.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor = None
        self.model = None
        if AutoProcessor is not None and AutoModel is not None:
            try:
                self.processor = AutoProcessor.from_pretrained(config.model_name)
            except Exception:
                try:
                    self.processor = AutoFeatureExtractor.from_pretrained(
                        config.model_name
                    )
                except Exception:
                    self.processor = None
            try:
                self.model = AutoModel.from_pretrained(config.model_name)
                self.model.to(self.device)
                self.model.eval()
            except Exception:
                self.model = None

    def encode(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        if not frames:
            return torch.zeros((1, 1), device=self.device)

        if self.model is None or self.processor is None:
            # Simple RGB mean descriptor fallback.
            stacked = torch.from_numpy(np.stack(frames)).float()
            pooled = stacked.mean(dim=(1, 2, 3)).unsqueeze(1)
            return pooled.to(self.device) / 255.0

        inputs = self.processor(images=list(frames), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        # Average the last hidden state if a pooler is unavailable.
        hidden = outputs.last_hidden_state
        return hidden.mean(dim=1)


class AudioEncoder:
    """Audio feature extractor built on top of ``torchaudio`` and ``transformers``."""

    def __init__(self, config: AudioEncoderConfig):
        self.config = config
        self.device = torch.device(
            config.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor = None
        self.model = None

        if AutoProcessor is not None and AutoModel is not None:
            try:
                self.processor = AutoProcessor.from_pretrained(config.model_name)
                self.model = AutoModel.from_pretrained(config.model_name)
                self.model.to(self.device)
                self.model.eval()
            except Exception:
                self.processor = None
                self.model = None

    def encode(self, video_path: str) -> torch.Tensor:
        waveform, sample_rate = self._load_audio(video_path)
        if waveform is None:
            return torch.zeros((1, 1), device=self.device)

        waveform = waveform.to(self.device)
        if self.processor is None or self.model is None:
            # Simple spectral energy summary.
            spec = torch.fft.rfft(waveform, dim=-1)
            energy = spec.abs().mean(dim=-1, keepdim=True)
            return energy

        inputs = self.processor(
            waveform, sampling_rate=sample_rate, return_tensors="pt"
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state
        return hidden.mean(dim=1)

    def _load_audio(self, video_path: str):
        if torchaudio is None or ffmpeg is None:
            return None, None

        try:
            process = (
                ffmpeg.input(video_path)
                .output("pipe:", format="wav", ac=1, ar=self.config.sample_rate)
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
        except Exception:
            return None, None

        audio_bytes = BytesIO(process[0])
        audio_bytes.seek(0)
        try:
            waveform, sample_rate = torchaudio.load(audio_bytes)
        except Exception:
            return None, None

        waveform = waveform.mean(dim=0, keepdim=True)
        return waveform, sample_rate

