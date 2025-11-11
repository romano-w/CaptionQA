"""Audio/visual encoder backbones for the captioning pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
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
    batch_size: int = 16
    cache_dir: Optional[str] = None
    use_cache: bool = True


@dataclass
class AudioEncoderConfig:
    """Configuration for the audio backbone."""

    model_name: str = "microsoft/wavlm-base-plus"
    sample_rate: int = 16000
    device: Optional[str] = None
    cache_dir: Optional[str] = None
    use_cache: bool = True


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
        self.cache_root = Path(self.config.cache_dir) if self.config.cache_dir else Path("data") / "cache" / "visual"
        self.cache_root.mkdir(parents=True, exist_ok=True)
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

    def encode(self, frames: Sequence[np.ndarray], *, cache_key: Optional[str] = None) -> torch.Tensor:
        if not frames:
            return torch.zeros((1, 1), device=self.device)

        # Try cache on CPU tensor for portability
        if self.config.use_cache and cache_key:
            cache_path = self.cache_root / f"{cache_key}.pt"
            if cache_path.exists():
                try:
                    return torch.load(cache_path, map_location=self.device)
                except Exception:
                    pass

        if self.model is None or self.processor is None:
            # Simple RGB mean descriptor fallback.
            stacked = torch.from_numpy(np.stack(frames)).float()
            pooled = stacked.mean(dim=(1, 2, 3)).unsqueeze(1)
            out = pooled.to(self.device) / 255.0
            if self.config.use_cache and cache_key:
                try:
                    torch.save(out.detach().cpu(), self.cache_root / f"{cache_key}.pt")
                except Exception:
                    pass
            return out

        # Batch the frames to avoid OOM
        bs = max(int(self.config.batch_size or 16), 1)
        embeddings = []
        for start in range(0, len(frames), bs):
            chunk = frames[start : start + bs]
            inputs = self.processor(images=list(chunk), return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            else:
                hidden = outputs.last_hidden_state
                emb = hidden.mean(dim=1)
            embeddings.append(emb)
        out = torch.cat(embeddings, dim=0)
        if self.config.use_cache and cache_key:
            try:
                torch.save(out.detach().cpu(), self.cache_root / f"{cache_key}.pt")
            except Exception:
                pass
        return out


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
        self.cache_root = Path(self.config.cache_dir) if self.config.cache_dir else Path("data") / "cache" / "audio"
        self.cache_root.mkdir(parents=True, exist_ok=True)

        if AutoProcessor is not None and AutoModel is not None:
            try:
                self.processor = AutoProcessor.from_pretrained(config.model_name)
                self.model = AutoModel.from_pretrained(config.model_name)
                self.model.to(self.device)
                self.model.eval()
            except Exception:
                self.processor = None
                self.model = None

    def encode(self, video_path: str, *, cache_key: Optional[str] = None) -> torch.Tensor:
        # Try cache first
        if self.config.use_cache and cache_key:
            cache_path = self.cache_root / f"{cache_key}.pt"
            if cache_path.exists():
                try:
                    return torch.load(cache_path, map_location=self.device)
                except Exception:
                    pass

        waveform, sample_rate = self._load_audio(video_path)
        if waveform is None:
            return torch.zeros((1, 1), device=self.device)

        waveform = waveform.to(self.device)
        if self.processor is None or self.model is None:
            # Simple spectral energy summary.
            spec = torch.fft.rfft(waveform, dim=-1)
            energy = spec.abs().mean(dim=-1, keepdim=True)
            out = energy
            if self.config.use_cache and cache_key:
                try:
                    torch.save(out.detach().cpu(), self.cache_root / f"{cache_key}.pt")
                except Exception:
                    pass
            return out

        inputs = self.processor(
            waveform, sampling_rate=sample_rate, return_tensors="pt"
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state
        out = hidden.mean(dim=1)
        if self.config.use_cache and cache_key:
            try:
                torch.save(out.detach().cpu(), self.cache_root / f"{cache_key}.pt")
            except Exception:
                pass
        return out

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

