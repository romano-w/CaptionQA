from pathlib import Path
import json

from captionqa.datasets.x360_manifest import build_manifest, merge_references, _write_jsonl, _read_json_or_jsonl


def test_manifest_build_and_merge(tmp_path: Path):
    # Create fake video files
    vids = [tmp_path / f"v{i}.mp4" for i in range(2)]
    for v in vids:
        v.parent.mkdir(parents=True, exist_ok=True)
        v.write_bytes(b"")

    rows = build_manifest(tmp_path, pattern="**/*.mp4")
    assert len(rows) == 2

    # Prepare refs
    refs_file = tmp_path / "refs.jsonl"
    _write_jsonl(refs_file, [{"id": vids[0].stem, "references": ["ref1"]}])
    merged = merge_references(rows, refs_file)
    id0 = vids[0].stem
    assert any(r.get("references") for r in merged if r["id"] == id0)

