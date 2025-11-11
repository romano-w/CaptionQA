"""Stub training loop for captioning soft-prompt fusion (MVP).

This script demonstrates how you might train the lightweight fusion head
and decoder soft prompt projection while keeping the LM frozen. It operates on
video paths + reference captions, using the feature encoders with caching.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Mapping

import torch

from .config import CaptioningConfig, build_pipeline


def _load_pairs(path: Path) -> List[Mapping[str, str]]:
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else [data]


def train(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train captioning fusion/soft prompt on small JSON pairs")
    p.add_argument("pairs_json", type=Path, help="JSON list of {video, caption}")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args(list(argv) if argv is not None else None)

    cfg = CaptioningConfig.from_defaults() if args.config is None else json.loads(Path(args.config).read_text())
    if isinstance(cfg, dict):
        # reuse CLI loader logic: create typed config
        from .cli import load_config

        cfg = load_config(args.config)

    sampler, venc, aenc, decoder, fusion = build_pipeline(cfg)
    device = torch.device(args.device)
    # Freeze LM parameters; train fusion + soft prompt projector if present
    if decoder.model is not None:
        for p_ in decoder.model.parameters():
            p_.requires_grad = False
    params = []
    if fusion.net is not None:
        params += list(fusion.net.parameters())
    # Projector is created lazily; create now with a dummy input
    if decoder.model is not None and decoder._cond_projector is None:
        dummy = torch.zeros(1, fusion.config.hidden_size)
        decoder.generate("warmup", conditioning=dummy)
    if decoder._cond_projector is not None:
        params += list(decoder._cond_projector.parameters())

    opt = torch.optim.AdamW(params, lr=args.lr) if params else None

    pairs = _load_pairs(args.pairs_json)
    for _ in range(args.epochs):
        for ex in pairs:
            video = ex["video"]
            target = ex["caption"].strip()
            frames = sampler.sample(video)
            vfeat = venc.encode(frames, cache_key=video)
            afeat = aenc.encode(video, cache_key=video + "|a")
            cond = fusion.fuse(vfeat, afeat)
            if decoder.model is None or opt is None:
                continue
            # Teacher forcing with LM loss: negative log-likelihood of target continuation
            tok = decoder.tokenizer
            if tok is None:
                continue
            tokens = tok(target, return_tensors="pt").to(device)
            input_embeds = decoder.model.get_input_embeddings()(tokens["input_ids"])  # type: ignore[index]
            # prepend soft prompt from conditioning
            prefix = decoder._cond_projector(cond.to(device))  # type: ignore[operator]
            E = decoder.model.get_input_embeddings().embedding_dim
            prefix = prefix.view(1, -1, E)
            inputs_embeds = torch.cat([prefix, input_embeds], dim=1)
            attention_mask = torch.ones(inputs_embeds.size()[:2], dtype=torch.long, device=device)
            outputs = decoder.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=tokens["input_ids"])  # type: ignore[index]
            loss = outputs.loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)


if __name__ == "__main__":  # pragma: no cover
    train()

