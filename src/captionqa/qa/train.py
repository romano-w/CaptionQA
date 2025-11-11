"""Minimal AVQA training script (dev subset friendly).

This trains the `AVQAModel` on a small subset by extracting features via the
shared encoders with caching enabled. It supports CPU/GPU, mixed precision, and
basic hyperparameters via CLI flags.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .datasets import AVQADataset
from .model import AVQAModel
from .tokenizer import SimpleWordTokenizer
from ..captioning.encoders import VisualEncoder, VisualEncoderConfig, AudioEncoder, AudioEncoderConfig
from ..captioning.panorama import PanoramicFrameSampler, PanoramaSamplingConfig


def _build_tokenizer(dataset: AVQADataset, save_path: Path | None = None) -> SimpleWordTokenizer:
    texts: List[str] = []
    for item in dataset:
        texts.append(item["question"])  # type: ignore[index]
        if item.get("answer"):
            texts.append(item["answer"])  # type: ignore[index]
    tok = SimpleWordTokenizer.build(texts)
    if save_path is not None:
        tok.save(save_path)
    return tok


def _collate(batch: List[Dict], sampler: PanoramicFrameSampler, venc: VisualEncoder, aenc: AudioEncoder, tok: SimpleWordTokenizer, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    videos: List[torch.Tensor] = []
    audios: List[torch.Tensor] = []
    q_tokens: List[List[int]] = []
    a_tokens: List[List[int]] = []

    for item in batch:
        video_path = str(item["video"])  # type: ignore[index]
        audio_path = str(item.get("audio") or item["video"])  # type: ignore[index]
        frames = sampler.sample(video_path)
        vfeat = venc.encode(frames, cache_key=video_path)
        if vfeat.dim() == 1:
            vfeat = vfeat.unsqueeze(0)
        videos.append(vfeat)

        afeat = aenc.encode(audio_path, cache_key=audio_path)
        if afeat.dim() == 1:
            afeat = afeat.unsqueeze(0)
        if afeat.dim() == 2:
            afeat = afeat.unsqueeze(1)
        audios.append(afeat)

        q_tokens.append(tok.encode(item["question"]))  # type: ignore[index]
        a_tokens.append(tok.encode(item.get("answer", "")))  # type: ignore[index]

    # Pad sequences to max lengths in batch
    def pad_list(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
        max_len = max(len(s) for s in seqs)
        out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    q = pad_list(q_tokens, tok.pad_token_id).to(device)
    a = pad_list(a_tokens, tok.pad_token_id).to(device)

    # For variable-length features, pad along time dimension to max in batch
    def pad_feats(feats: List[torch.Tensor]) -> torch.Tensor:
        max_t = max(f.size(0) for f in feats)
        dim = feats[0].size(-1)
        out = torch.zeros((len(feats), max_t, dim))
        for i, f in enumerate(feats):
            out[i, : f.size(0), : dim] = f[:, :dim]
        return out.to(device)

    v = pad_feats(videos)
    a_feat = pad_feats(audios)
    return v, a_feat, q, a


def train(argv: Iterable[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description="Train AVQAModel on a small subset using cached features")
    p.add_argument("dataset_root", type=Path)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--tokenizer", type=Path, default=None, help="Path to save/load tokenizer JSON")
    p.add_argument("--output", type=Path, default=Path("checkpoints/avqa_tiny.pt"))
    args = p.parse_args(list(argv) if argv is not None else None)

    device = torch.device(args.device)
    ds = AVQADataset(args.dataset_root)
    # Tiny subset for quick training
    if len(ds) > 16:
        ds.samples = ds.samples[:16]  # type: ignore[attr-defined]

    # Tokenizer
    if args.tokenizer and args.tokenizer.exists():
        tok = SimpleWordTokenizer.load(args.tokenizer)
    else:
        tok = _build_tokenizer(ds, save_path=args.tokenizer)

    # Feature pipeline
    sampler = PanoramicFrameSampler(PanoramaSamplingConfig())
    venc = VisualEncoder(VisualEncoderConfig())
    aenc = AudioEncoder(AudioEncoderConfig())

    # Infer dims from first example
    first = ds[0]
    v0, a0, _, _ = _collate([first], sampler, venc, aenc, tok, device)
    video_dim = v0.size(-1)
    audio_dim = a0.size(-1)
    model = AVQAModel(video_dim=video_dim, audio_dim=audio_dim, vocab_size=len(tok.itos)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: _collate(b, sampler, venc, aenc, tok, device))
    model.train()
    for epoch in range(args.epochs):
        for batch in loader:
            v, a_feat, q, a = batch
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                out = model(v, a_feat, q, answer_tokens=a)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

    # Save minimal checkpoint
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "config": {
            "video_dim": video_dim,
            "audio_dim": audio_dim,
            "vocab_size": len(tok.itos),
        },
    }
    torch.save(state, args.output)
    return args.output


if __name__ == "__main__":  # pragma: no cover
    train()

