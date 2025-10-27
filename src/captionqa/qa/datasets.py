"""Dataset loaders for multimodal question answering benchmarks.

This module currently focuses on the AVQA benchmark which contains
panoramic (360°) video clips with corresponding binaural spatial audio
and textual question/answer annotations.  The dataset loader aligns the
three modalities so downstream pipelines receive tuples containing:

* the natural language question (string)
* the spatially-aware panoramic video (path or transformed tensor)
* the binaural audio clip (path or transformed tensor)
* optional temporal localization metadata (start/end timestamps)

The implementation is intentionally lightweight and does not depend on
the actual media codecs.  Callers can inject custom transforms for
video/audio decoding or feature extraction while the loader handles path
resolution and annotation parsing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

try:  # pragma: no cover - torch is optional for documentation builds
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - fall back to a Sequence-like base class
    class Dataset(Sequence):  # type: ignore[misc]
        """Minimal stand-in for ``torch.utils.data.Dataset``.

        The fallback only aims to provide the ``Sequence`` protocol so the
        class can be used for simple iteration in environments where PyTorch
        is not installed (e.g., documentation builds).
        """

        def __getitem__(self, index: int) -> Any:  # pragma: no cover - interface stub
            raise NotImplementedError

        def __len__(self) -> int:  # pragma: no cover - interface stub
            raise NotImplementedError


Annotation = Dict[str, Any]
PathLike = Union[str, os.PathLike[str]]


@dataclass(slots=True)
class AVQASample:
    """Container for a single AVQA triplet.

    Parameters
    ----------
    question_id:
        Unique identifier for the question/answer pair.
    question:
        Natural language question string.
    answer:
        Ground-truth answer string.  May be ``None`` for test splits.
    video_path:
        Absolute path to the panoramic video clip aligned with the
        question.
    audio_path:
        Absolute path to the binaural audio clip corresponding to the
        question.
    start_time, end_time:
        Optional temporal window (in seconds) delimiting the region of
        interest that contains the answer evidence.
    metadata:
        Additional annotation payload preserved verbatim for advanced use
        cases (e.g., confidence scores, answer types, etc.).
    """

    question_id: str
    question: str
    answer: Optional[str]
    video_path: Path
    audio_path: Path
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def temporal_window(self) -> Optional[Tuple[float, float]]:
        """Return the temporal window, if both bounds are available."""

        if self.start_time is None or self.end_time is None:
            return None
        return (self.start_time, self.end_time)


def _coerce_path(root: Path, value: str, default_dir: Optional[str]) -> Path:
    """Resolve ``value`` to an absolute :class:`~pathlib.Path`.

    ``value`` may already be absolute.  Otherwise, ``root / value`` is
    attempted followed by ``root / default_dir / value`` when
    ``default_dir`` is provided.  The function does not require the path
    to exist; many development flows operate on placeholder files until
    data is synced locally.
    """

    candidate = Path(value)
    if candidate.is_absolute():
        return candidate

    direct = root / candidate
    if direct.exists():
        return direct

    if default_dir is not None:
        nested = root / default_dir / candidate
        if nested.exists():
            return nested
        return nested

    return direct


def _normalise_annotation_list(payload: Any) -> List[Annotation]:
    """Extract a flat list of annotations from an arbitrary JSON payload."""

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("data", "annotations", "questions", "items", "records"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
        # Fall back to treating values as the records (e.g. mapping of id -> item)
        if all(isinstance(v, dict) for v in payload.values()):
            return [dict(v, question_id=k) for k, v in payload.items()]

    raise ValueError("Unsupported AVQA annotation format")


def _extract_annotation_fields(annotation: Annotation) -> Tuple[str, str, Optional[str], str, str, Optional[float], Optional[float], Dict[str, Any]]:
    """Extract core fields from a raw annotation dictionary."""

    question_id = str(annotation.get("question_id") or annotation.get("id") or annotation.get("qid"))
    if not question_id:
        raise KeyError("Annotation is missing a question identifier")

    question = annotation.get("question")
    if not isinstance(question, str):
        raise KeyError(f"Annotation {question_id} is missing 'question'")

    answer = annotation.get("answer")
    if answer is not None and not isinstance(answer, str):
        answer = str(answer)

    video_key = annotation.get("video") or annotation.get("video_path") or annotation.get("video_name")
    if not isinstance(video_key, str):
        raise KeyError(f"Annotation {question_id} is missing a video reference")

    audio_key = annotation.get("audio") or annotation.get("audio_path") or annotation.get("audio_name") or annotation.get("binaural_audio")
    if not isinstance(audio_key, str):
        # Some AVQA annotations reuse the video filename for audio assets.
        # In that case we fall back to the video key and let the caller
        # adjust directories via ``audio_dir``.
        audio_key = video_key

    start = annotation.get("start_time") or annotation.get("start") or annotation.get("temporal_start")
    end = annotation.get("end_time") or annotation.get("end") or annotation.get("temporal_end")

    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive programming
            raise ValueError(f"Cannot convert {value!r} to float") from exc

    start_f = _coerce_float(start)
    end_f = _coerce_float(end)

    preserved = dict(annotation)
    preserved.pop("question", None)
    preserved.pop("answer", None)
    preserved.pop("video", None)
    preserved.pop("video_path", None)
    preserved.pop("video_name", None)
    preserved.pop("audio", None)
    preserved.pop("audio_path", None)
    preserved.pop("audio_name", None)
    preserved.pop("binaural_audio", None)
    preserved.pop("start_time", None)
    preserved.pop("start", None)
    preserved.pop("temporal_start", None)
    preserved.pop("end_time", None)
    preserved.pop("end", None)
    preserved.pop("temporal_end", None)

    return question_id, question, answer, video_key, audio_key, start_f, end_f, preserved


class AVQADataset(Dataset):
    """Iterate over aligned AVQA question/video/audio tuples.

    Parameters
    ----------
    root:
        Root directory of the AVQA dataset (i.e., the folder containing
        ``annotations/`` and media sub-directories).
    split:
        Dataset split to load (e.g., ``"train"``, ``"val"``, ``"test"``).
    annotation_file:
        Optional custom annotation JSON path.  When ``None``, a handful of
        conventional filenames are probed automatically.
    video_dir, audio_dir:
        Relative directories (under ``root``) that contain video and audio
        assets.  Override these if the dataset is organised differently or
        if annotations use bare filenames without folder prefixes.
    question_transform, video_transform, audio_transform, sample_transform:
        Callable hooks applied during ``__getitem__`` to customise loaded
        items.  ``sample_transform`` receives the fully assembled sample
        dictionary and is invoked last.
    """

    def __init__(
        self,
        root: PathLike,
        *,
        split: str = "train",
        annotation_file: Optional[PathLike] = None,
        video_dir: Optional[str] = "videos",
        audio_dir: Optional[str] = "audio",
        question_transform: Optional[Callable[[str], Any]] = None,
        video_transform: Optional[Callable[[Path], Any]] = None,
        audio_transform: Optional[Callable[[Path], Any]] = None,
        sample_transform: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.question_transform = question_transform
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.sample_transform = sample_transform

        annotation_path = self._resolve_annotation_file(annotation_file)
        annotations = self._load_annotations(annotation_path)
        self.samples: List[AVQASample] = annotations

    # ------------------------------------------------------------------
    # Helper API
    # ------------------------------------------------------------------
    def _resolve_annotation_file(self, custom: Optional[PathLike]) -> Path:
        if custom is not None:
            path = Path(custom)
            if not path.is_absolute():
                path = self.root / path
            if not path.exists():
                raise FileNotFoundError(f"AVQA annotations not found: {path}")
            return path

        candidates = [
            self.root / "annotations" / f"{self.split}.json",
            self.root / "annotations" / f"{self.split}_qa.json",
            self.root / f"{self.split}.json",
            self.root / f"{self.split}_qa.json",
        ]

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"Could not locate annotations for split '{self.split}'. "
            "Provide `annotation_file` explicitly."
        )

    def _load_annotations(self, annotation_path: Path) -> List[AVQASample]:
        with annotation_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)

        entries = _normalise_annotation_list(payload)
        samples: List[AVQASample] = []
        for record in entries:
            (
                question_id,
                question,
                answer,
                video_key,
                audio_key,
                start_time,
                end_time,
                metadata,
            ) = _extract_annotation_fields(record)

            video_path = _coerce_path(self.root, video_key, self.video_dir)
            audio_path = _coerce_path(self.root, audio_key, self.audio_dir)

            samples.append(
                AVQASample(
                    question_id=question_id,
                    question=question,
                    answer=answer,
                    video_path=video_path,
                    audio_path=audio_path,
                    start_time=start_time,
                    end_time=end_time,
                    metadata=metadata,
                )
            )

        return samples

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]

        question = sample.question
        if self.question_transform is not None:
            question = self.question_transform(question)

        video = sample.video_path
        if self.video_transform is not None:
            video = self.video_transform(video)

        audio = sample.audio_path
        if self.audio_transform is not None:
            audio = self.audio_transform(audio)

        result: Dict[str, Any] = {
            "question_id": sample.question_id,
            "question": question,
            "answer": sample.answer,
            "video": video,
            "audio": audio,
            "temporal_window": sample.temporal_window(),
            "metadata": sample.metadata,
        }

        if self.sample_transform is not None:
            return self.sample_transform(result)

        return result

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @classmethod
    def discover_root(cls, base: Optional[PathLike] = None) -> Path:
        """Attempt to discover the AVQA dataset root.

        The search order is:

        1. ``base`` (if provided)
        2. ``$CAPTIONQA_DATASETS/avqa/AVQA``
        3. ``$CAPTIONQA_DATASETS/AVQA``

        The environment variable mirrors the documentation examples and
        provides a central location to toggle between workstations.
        """

        if base is not None:
            path = Path(base).expanduser()
            if path.exists():
                return path

        env_root = os.environ.get("CAPTIONQA_DATASETS")
        if env_root:
            candidates = [
                Path(env_root) / "avqa" / "AVQA",
                Path(env_root) / "AVQA",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate

        raise FileNotFoundError(
            "Unable to locate the AVQA dataset root. Set the "
            "CAPTIONQA_DATASETS environment variable or pass an explicit path."
        )

    def iter_questions(self) -> Iterator[str]:
        """Iterate over raw questions for quick text-only experiments."""

        for sample in self.samples:
            yield sample.question

    def iter_media_pairs(self) -> Iterator[Tuple[Path, Path]]:
        """Iterate over (video_path, audio_path) tuples."""

        for sample in self.samples:
            yield sample.video_path, sample.audio_path


__all__ = ["AVQADataset", "AVQASample"]

