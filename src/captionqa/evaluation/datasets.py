"""Wrappers around Hugging Face datasets for evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

from datasets import Dataset, load_dataset

from .metrics import normalize_references


@dataclass
class CaptionDatasetConfig:
    """Configuration for loading caption references from a dataset."""

    name: str
    split: str
    config: Optional[str] = None
    id_column: str = "id"
    reference_column: str = "references"

    def load(self) -> Mapping[str, List[str]]:
        dataset = self._load_split()
        references: Dict[str, List[str]] = {}
        for row in dataset:
            example_id = str(row[self.id_column])
            values = row[self.reference_column]
            if isinstance(values, str):
                references[example_id] = [values]
            else:
                references[example_id] = normalize_references(values)
        return references

    def _load_split(self) -> Dataset:
        if self.config is None:
            return load_dataset(self.name, split=self.split)
        return load_dataset(self.name, self.config, split=self.split)


@dataclass
class QADatasetConfig:
    """Configuration for loading QA references from a dataset."""

    name: str
    split: str
    config: Optional[str] = None
    id_column: str = "id"
    answer_column: str = "answers"

    def load(self) -> Mapping[str, List[str]]:
        dataset = self._load_split()
        answers: Dict[str, List[str]] = {}
        for row in dataset:
            example_id = str(row[self.id_column])
            values = row[self.answer_column]
            if isinstance(values, str):
                answers[example_id] = [values]
            else:
                answers[example_id] = normalize_references(values)
        return answers

    def _load_split(self) -> Dataset:
        if self.config is None:
            return load_dataset(self.name, split=self.split)
        return load_dataset(self.name, self.config, split=self.split)


def dataset_summary(name: str, split: str, num_examples: int, columns: Iterable[str]) -> Dict[str, str]:
    """Create a lightweight summary dictionary for logging or JSON output."""

    return {
        "dataset": name,
        "split": split,
        "num_examples": num_examples,
        "columns": ", ".join(columns),
    }
