"""Helpers to normalize free-form QA predictions to TAL action labels."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List


def _compile(patterns: Iterable[str]) -> List[re.Pattern[str]]:
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns]


TAL_LABELS: List[str] = [
    "walking",
    "operating phone",
    "dressing",
    "speaking",
    "opening",
    "sitting",
    "coughing",
    "dancing",
    "drinking",
    "housekeeping",
    "pouring",
    "eating",
    "playing",
    "cleaning",
    "photographing",
    "standing",
    "workout",
    "laughing",
    "farming",
]


LABEL_PATTERNS: Dict[str, List[re.Pattern[str]]] = {
    "walking": _compile(
        [
            r"\bwalk\w*\b",
            r"\bstroll\w*\b",
            r"\bpace\w*\b",
            r"\bwandering\b",
            r"\bmoving around\b",
            r"\bmove(?:s|d)? around\b",
            r"\bwalks toward\b",
            r"\bwalk(?:ing)? around the room\b",
        ]
    ),
    "operating phone": _compile(
        [
            r"\bcell(?: )?phone\b",
            r"\bsmart(?: )?phone\b",
            r"\bmobile\b",
            r"\bphone\b",
            r"\btext(?:ing|s|ed)?\b",
            r"\bscroll(?:ing|s|ed)?\b",
            r"\btyping on (?:a )?phone\b",
            r"\busing (?:a )?phone\b",
            r"\bholding (?:a )?(?:phone|cell)\b",
            r"\bselfie\b",
        ]
    ),
    "dressing": _compile(
        [
            r"\bdress(?:ing)?\b",
            r"\bput(?:ting)? on\b",
            r"\bchang(?:ing)? (?:clothes|shirt|outfit)\b",
            r"\bbutton\w*\b",
            r"\bzip(?:ping)?\b",
        ]
    ),
    "speaking": _compile(
        [
            r"\bspeak\w*\b",
            r"\btalk\w*\b",
            r"\bconversation\b",
            r"\baddress\w*\b",
            r"\bchat\w*\b",
            r"\binterview\w*\b",
            r"\bresponds?\b",
            r"\bmicrophone\b",
            r"\bvoice\b",
            r"\bannounc\w*\b",
            r"\bspeech\b",
        ]
    ),
    "opening": _compile(
        [
            r"\bopen\w*\b",
            r"\bunbox\w*\b",
            r"\bunpack\w*\b",
            r"\bunzip\w*\b",
            r"\bopens? (?:the )?(?:door|window|fridge|drawer|cabinet)\b",
        ]
    ),
    "sitting": _compile(
        [
            r"\bsit(?:ting|s)?\b",
            r"\bsat\b",
            r"\bseated\b",
            r"\btakes a seat\b",
            r"\bremain seated\b",
            r"\bsits down\b",
        ]
    ),
    "coughing": _compile([r"\bcough\w*\b", r"\bclears? (?:his|her) throat\b"]),
    "dancing": _compile([r"\bdanc\w*\b"]),
    "drinking": _compile(
        [
            r"\bdrink\w*\b",
            r"\bsip\w*\b",
            r"\bgulp\w*\b",
            r"\bholding (?:a|the)? (?:cup|glass|mug|bottle)\b",
            r"\bcoffee\b",
            r"\btea\b",
            r"\bbeverage\b",
        ]
    ),
    "housekeeping": _compile(
        [
            r"\bhousekeep\w*\b",
            r"\bhouse\s*work\b",
            r"\bchores?\b",
            r"\bmake\w*\b.*\bbed\b",
            r"\btidy\w*\b",
            r"\borganiz\w*\b",
        ]
    ),
    "pouring": _compile([r"\bpour\w*\b", r"\bfill\w*\b.*\b(glass|cup|bottle)\b", r"\bpick\w* up (?:a )?(?:bottle|kettle)\b"]),
    "eating": _compile(
        [
            r"\beat\w*\b",
            r"\bbite\w*\b",
            r"\bchew\w*\b",
            r"\bmunch\w*\b",
            r"\bhold\w* (?:a |the )?(?:plate|bowl)\b",
            r"\bserve\w* food\b",
            r"\bstirr\w* (?:food|pot|soup)\b",
            r"\bcut\w* (?:food|meat)\b",
        ]
    ),
    "playing": _compile([r"\bplay\w*\b", r"\bgaming\b", r"\binstrument\b", r"\bguitar\b", r"\bviolin\b", r"\bpiano\b"]),
    "cleaning": _compile(
        [r"\bclean\w*\b", r"\bwip\w*\b", r"\bsweep\w*\b", r"\bmop\w*\b", r"\bscrub\w*\b", r"\bdust\w*\b"]
    ),
    "photographing": _compile(
        [
            r"\bphotograph\w*\b",
            r"\bphoto\b",
            r"\bpictur\w*\b",
            r"\bselfie\b",
            r"\bcamera\b",
            r"\bshoot\w*\b",
            r"\bfilming\b",
            r"\bvideotap\w*\b",
            r"\brecord\w* (?:a )?(?:video|scene)\b",
        ]
    ),
    "standing": _compile([r"\bstand\w*\b", r"\bstood\b", r"\bremain standing\b"]),
    "workout": _compile(
        [
            r"\bwork\s*out\b",
            r"\bworkout\b",
            r"\bexercise\w*\b",
            r"\bpush-?ups?\b",
            r"\bsit-?ups?\b",
            r"\blift\w*\b",
            r"\btraining\b",
            r"\bjog\w*\b",
            r"\byoga\b",
        ]
    ),
    "laughing": _compile([r"\blaugh\w*\b", r"\bgiggl\w*\b"]),
    "farming": _compile([r"\bfarm\w*\b", r"\bharvest\w*\b", r"\bplow\w*\b", r"\btractor\b", r"\bcrop\w*\b"]),
}

LABEL_PRIORITY = TAL_LABELS


def normalize_prediction(text: str) -> str:
    if not text:
        return text
    lowered = text.lower()
    for label in LABEL_PRIORITY:
        for pattern in LABEL_PATTERNS[label]:
            if pattern.search(lowered):
                return label
    return text.strip()


__all__ = ["normalize_prediction", "TAL_LABELS"]
