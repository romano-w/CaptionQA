"""Caption decoder heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

try:  # pragma: no cover - optional heavy dependency
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - keep CLI functional without transformers
    AutoModelForCausalLM = None
    AutoTokenizer = None


@dataclass
class CaptionDecoderConfig:
    """Configuration for the caption decoder head."""

    model_name: str = "hf-internal-testing/tiny-random-gpt2"
    device: Optional[str] = None
    max_new_tokens: int = 64


class CaptionDecoder:
    """Wraps a language model to autoregressively generate captions."""

    def __init__(self, config: CaptionDecoderConfig):
        self.config = config
        self.device = torch.device(
            config.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = None
        self.model = None

        if AutoTokenizer is not None and AutoModelForCausalLM is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.model_name
                )
                self.model.to(self.device)
                self.model.eval()
            except Exception:
                self.model = None
                self.tokenizer = None

    def generate(self, prompt: str) -> str:
        """Generate a caption conditioned on ``prompt``."""

        if not prompt:
            prompt = "Describe the video contents succinctly."

        if self.model is None or self.tokenizer is None:
            # Deterministic fallback when transformers aren't available.
            return f"{prompt.strip()} [Caption generation requires transformers models.]"

        tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                **tokens,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

