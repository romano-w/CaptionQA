"""Command line interface for panoramic caption generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from . import CaptioningConfig, generate_captions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate captions for 360° panoramic videos."
    )
    parser.add_argument("video", type=Path, help="Path to the video file")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the salient events occurring in this 360-degree video.",
        help="Custom prompt guiding caption generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override the decoder's max_new_tokens parameter.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON configuration overriding component defaults.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved configuration before running the pipeline.",
    )
    return parser


def load_config(path: Optional[Path]) -> CaptioningConfig:
    if path is None:
        return CaptioningConfig.from_defaults()

    data = json.loads(path.read_text())

    panorama = data.get("panorama", {})
    visual = data.get("visual_encoder", {})
    audio = data.get("audio_encoder", {})
    decoder = data.get("decoder", {})

    defaults = CaptioningConfig.from_defaults()
    return CaptioningConfig.from_defaults(
        panorama=defaults.panorama.__class__(**{**defaults.panorama.__dict__, **panorama}),
        visual_encoder=defaults.visual_encoder.__class__(
            **{**defaults.visual_encoder.__dict__, **visual}
        ),
        audio_encoder=defaults.audio_encoder.__class__(
            **{**defaults.audio_encoder.__dict__, **audio}
        ),
        decoder=defaults.decoder.__class__(
            **{**defaults.decoder.__dict__, **decoder}
        ),
    )


def main(argv: Optional[List[str]] = None) -> str:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.video.exists():
        raise FileNotFoundError(args.video)

    config = load_config(args.config)
    if args.print_config:
        print(json.dumps(config, default=lambda o: o.__dict__, indent=2))

    caption = generate_captions(
        str(args.video),
        config=config,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )
    print(caption)
    return caption


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

