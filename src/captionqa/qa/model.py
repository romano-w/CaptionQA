"""Multimodal fusion modules and AVQA question answering model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class VideoAudioCrossAttention(nn.Module):
    """Bidirectional cross-attention between video patches and audio features."""

    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_to_audio = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=False)
        self.audio_to_video = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=False)
        self.video_norm = nn.LayerNorm(hidden_dim)
        self.audio_norm = nn.LayerNorm(hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        video: Tensor,
        audio: Tensor,
        *,
        video_mask: Optional[Tensor] = None,
        audio_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Fuse video/audio streams via cross-attention.

        Parameters
        ----------
        video:
            Tensor of shape ``(batch, frames, video_dim)`` representing
            patch-level video descriptors.
        audio:
            Tensor of shape ``(batch, spectrogram_frames, audio_dim)``
            containing binaural audio features.
        video_mask, audio_mask:
            Boolean masks (``True`` denotes padding) to exclude padded
            elements when attending across modalities.
        """

        video_proj = self.video_proj(video)
        audio_proj = self.audio_proj(audio)

        v = video_proj.transpose(0, 1)  # (frames, batch, hidden)
        a = audio_proj.transpose(0, 1)

        v_mask = video_mask
        a_mask = audio_mask

        cross_v, _ = self.video_to_audio(v, a, a, key_padding_mask=a_mask)
        cross_a, _ = self.audio_to_video(a, v, v, key_padding_mask=v_mask)

        fused_video = self.video_norm((cross_v + v).transpose(0, 1))
        fused_audio = self.audio_norm((cross_a + a).transpose(0, 1))

        pooled = self.fusion(
            torch.cat(
                [fused_video.mean(dim=1), fused_audio.mean(dim=1)], dim=-1
            )
        )

        return {"video": fused_video, "audio": fused_audio, "pooled": pooled}


class QuestionEncoder(nn.Module):
    """Encode tokenised questions into a single vector representation."""

    def __init__(self, vocab_size: int, hidden_dim: int, pad_token_id: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.pad_token_id = pad_token_id

    def forward(self, tokens: Tensor) -> Tensor:
        emb = self.embedding(tokens)
        output, _ = self.gru(emb)
        mask = (tokens != self.pad_token_id).unsqueeze(-1)
        masked_output = output * mask
        denom = mask.sum(dim=1).clamp_min(1)
        return masked_output.sum(dim=1) / denom


class AnswerDecoder(nn.Module):
    """Autoregressive decoder conditioned on multimodal context."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.num_layers = num_layers

    def init_hidden(self, context: Tensor) -> Tensor:
        expanded = context.unsqueeze(0).repeat(self.num_layers, 1, 1)
        return expanded.contiguous()

    def forward(self, tokens: Tensor, context: Tensor) -> Tuple[Tensor, Tensor]:
        hidden = self.init_hidden(context)
        emb = self.embedding(tokens)
        output, hidden = self.gru(emb, hidden)
        logits = self.output(output)
        return logits, hidden

    def greedy_decode(self, context: Tensor, max_length: int) -> Tensor:
        hidden = self.init_hidden(context)
        batch = hidden.size(1)
        inputs = torch.full((batch, 1), self.bos_token_id, dtype=torch.long, device=hidden.device)
        decoded: List[Tensor] = []
        states = hidden
        for _ in range(max_length):
            emb = self.embedding(inputs)
            output, states = self.gru(emb, states)
            logits = self.output(output)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            decoded.append(next_token)
            inputs = next_token
            if torch.all(next_token.squeeze(-1) == self.eos_token_id):
                break
        if decoded:
            return torch.cat(decoded, dim=1)
        return torch.empty((batch, 0), dtype=torch.long, device=hidden.device)

    def greedy_decode_with_confidence(self, context: Tensor, max_length: int) -> Tuple[Tensor, Tensor]:
        """Greedy decode returning token IDs and per-step max probabilities.

        Returns
        -------
        tokens: LongTensor
            Shape ``(batch, seq_len)`` of generated token ids.
        confidences: FloatTensor
            Shape ``(batch, seq_len)`` of max softmax probabilities for each step.
        """
        hidden = self.init_hidden(context)
        batch = hidden.size(1)
        inputs = torch.full((batch, 1), self.bos_token_id, dtype=torch.long, device=hidden.device)
        decoded: List[Tensor] = []
        confs: List[Tensor] = []
        states = hidden
        for _ in range(max_length):
            emb = self.embedding(inputs)
            output, states = self.gru(emb, states)
            logits = self.output(output)
            probs = logits[:, -1, :].softmax(dim=-1)
            max_prob, next_token = probs.max(dim=-1, keepdim=True)
            decoded.append(next_token)
            confs.append(max_prob)
            inputs = next_token
            if torch.all(next_token.squeeze(-1) == self.eos_token_id):
                break
        if decoded:
            return torch.cat(decoded, dim=1), torch.cat(confs, dim=1)
        return (
            torch.empty((batch, 0), dtype=torch.long, device=hidden.device),
            torch.empty((batch, 0), dtype=torch.float, device=hidden.device),
        )


class AVQAModel(nn.Module):
    """End-to-end multimodal question answering network for AVQA."""

    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        vocab_size: int,
        *,
        hidden_dim: int = 512,
        num_heads: int = 4,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> None:
        super().__init__()
        self.fusion = VideoAudioCrossAttention(video_dim, audio_dim, hidden_dim, num_heads=num_heads)
        self.question_encoder = QuestionEncoder(vocab_size, hidden_dim, pad_token_id)
        self.answer_decoder = AnswerDecoder(
            vocab_size,
            hidden_dim,
            pad_token_id,
            bos_token_id,
            eos_token_id,
        )
        self.context_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.default_max_answer_len = 16

    def forward(
        self,
        video: Tensor,
        audio: Tensor,
        question_tokens: Tensor,
        answer_tokens: Optional[Tensor] = None,
        *,
        video_mask: Optional[Tensor] = None,
        audio_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        fusion = self.fusion(video, audio, video_mask=video_mask, audio_mask=audio_mask)
        question_repr = self.question_encoder(question_tokens)
        context = self.context_proj(torch.cat([fusion["pooled"], question_repr], dim=-1))

        outputs: Dict[str, Tensor] = {}

        if answer_tokens is not None:
            decoder_input = answer_tokens[:, :-1]
            target = answer_tokens[:, 1:]
            logits, _ = self.answer_decoder(decoder_input, context)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=self.pad_token_id,
            )
            outputs["loss"] = loss
            outputs["logits"] = logits
        else:
            generated = self.answer_decoder.greedy_decode(
                context, max_length=self.default_max_answer_len
            )
            outputs["generated_tokens"] = generated

        return outputs

    @torch.no_grad()
    def generate(
        self,
        video: Tensor,
        audio: Tensor,
        question_tokens: Tensor,
        *,
        video_mask: Optional[Tensor] = None,
        audio_mask: Optional[Tensor] = None,
        max_length: int = 16,
    ) -> Tensor:
        fusion = self.fusion(video, audio, video_mask=video_mask, audio_mask=audio_mask)
        question_repr = self.question_encoder(question_tokens)
        context = self.context_proj(torch.cat([fusion["pooled"], question_repr], dim=-1))
        decoded = self.answer_decoder.greedy_decode(context, max_length=max_length)
        return decoded

    @torch.no_grad()
    def generate_with_confidence(
        self,
        video: Tensor,
        audio: Tensor,
        question_tokens: Tensor,
        *,
        video_mask: Optional[Tensor] = None,
        audio_mask: Optional[Tensor] = None,
        max_length: int = 16,
    ) -> Tuple[Tensor, Tensor]:
        fusion = self.fusion(video, audio, video_mask=video_mask, audio_mask=audio_mask)
        question_repr = self.question_encoder(question_tokens)
        context = self.context_proj(torch.cat([fusion["pooled"], question_repr], dim=-1))
        tokens, probs = self.answer_decoder.greedy_decode_with_confidence(context, max_length=max_length)
        return tokens, probs


@dataclass
class GeneratedAnswer:
    tokens: List[int]
    text: str


def decode_answers(tokenizer: Any, sequences: Tensor) -> List[GeneratedAnswer]:
    """Convert generated token IDs into human-readable strings."""

    results: List[GeneratedAnswer] = []
    for seq in sequences.tolist():
        if hasattr(tokenizer, "decode"):
            text = tokenizer.decode(seq, skip_special_tokens=True)
        else:
            # Fallback: join tokens if tokenizer lacks decode API.
            text = " ".join(str(token) for token in seq)
        results.append(GeneratedAnswer(tokens=seq, text=text))
    return results


__all__ = [
    "AVQAModel",
    "VideoAudioCrossAttention",
    "QuestionEncoder",
    "AnswerDecoder",
    "decode_answers",
    "GeneratedAnswer",
]

