"""A minimal whitespace tokenizer for AVQA with serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


@dataclass
class SimpleWordTokenizer:
    stoi: Dict[str, int]
    itos: List[str]
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1) -> "SimpleWordTokenizer":
        counts: Dict[str, int] = {}
        for t in texts:
            for w in t.strip().lower().split():
                counts[w] = counts.get(w, 0) + 1

        vocab = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
        vocab += [w for w, c in sorted(counts.items()) if c >= min_freq and w not in vocab]
        stoi = {w: i for i, w in enumerate(vocab)}
        itos = vocab
        return cls(stoi=stoi, itos=itos, pad_token_id=stoi[PAD_TOKEN], bos_token_id=stoi[BOS_TOKEN], eos_token_id=stoi[EOS_TOKEN])

    def encode(self, text: str, add_special_tokens: bool = True, max_length: int | None = None) -> List[int]:
        ids: List[int] = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        for w in text.strip().lower().split():
            ids.append(self.stoi.get(w, self.stoi[UNK_TOKEN]))
        if add_special_tokens:
            ids.append(self.eos_token_id)
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        words: List[str] = []
        specials = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        for i in ids:
            if skip_special_tokens and i in specials:
                continue
            if 0 <= i < len(self.itos):
                words.append(self.itos[i])
        return " ".join(words)

    def save(self, path: Path) -> None:
        payload = {
            "itos": self.itos,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: Path) -> "SimpleWordTokenizer":
        data = json.loads(path.read_text())
        itos = list(data["itos"])
        stoi = {w: i for i, w in enumerate(itos)}
        return cls(stoi=stoi, itos=itos, pad_token_id=data["pad_token_id"], bos_token_id=data["bos_token_id"], eos_token_id=data["eos_token_id"])


__all__ = ["SimpleWordTokenizer", "PAD_TOKEN", "UNK_TOKEN", "BOS_TOKEN", "EOS_TOKEN"]

