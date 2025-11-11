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
    soft_prompt_tokens: int = 4


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
        self._cond_projector = None  # lazily initialized when conditioning arrives

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

    def generate(self, prompt: str, conditioning: Optional[torch.Tensor] = None) -> str:
        """Generate a caption conditioned on ``prompt``."""

        if not prompt:
            prompt = "Describe the video contents succinctly."

        if self.model is None or self.tokenizer is None:
            # Deterministic fallback when transformers aren't available.
            return f"{prompt.strip()} [Caption generation requires transformers models.]"

        # Standard text-only path if no conditioning provided
        if conditioning is None:
            tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated = self.model.generate(
                    **tokens,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                )
            return self.tokenizer.decode(generated[0], skip_special_tokens=True)

        # Multimodal path: project conditioning to a soft prompt prefix
        cond = conditioning.detach().to(self.device)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)

        # Lazily create projector to match model embedding size
        embed_dim = self.model.get_input_embeddings().embedding_dim
        prompt_len = max(int(self.config.soft_prompt_tokens or 4), 1)
        if self._cond_projector is None or (
            self._cond_projector.out_features != embed_dim * prompt_len
        ):
            in_dim = cond.shape[-1]
            self._cond_projector = torch.nn.Linear(in_dim, embed_dim * prompt_len).to(
                self.device
            )
            # initialize small to avoid overwhelming prompt
            torch.nn.init.xavier_uniform_(self._cond_projector.weight, gain=0.1)
            if self._cond_projector.bias is not None:
                torch.nn.init.zeros_(self._cond_projector.bias)

        prefix = self._cond_projector(cond)  # [B, embed_dim * prompt_len]
        prefix = prefix.view(prefix.size(0), prompt_len, embed_dim)

        tok = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_embeds = self.model.get_input_embeddings()(tok["input_ids"])  # [B, T, E]
        inputs_embeds = torch.cat([prefix, input_embeds], dim=1)
        attention_mask = torch.ones(
            (inputs_embeds.size(0), inputs_embeds.size(1)), dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            try:
                generated = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                )
            except Exception:
                # Fall back to text-only path if model doesn't support inputs_embeds
                tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                generated = self.model.generate(
                    **tokens,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

