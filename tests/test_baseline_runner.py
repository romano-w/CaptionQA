from pathlib import Path
import json

from captionqa.captioning.baseline import run


def test_baseline_with_manifest_and_refs(tmp_path: Path):
    # Prepare manifest pointing to the existing dummy clip and a refs file
    video = Path("data/dev-mini/samples/dummy.mp4")
    assert video.exists(), "dev-mini dummy clip should exist"

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": "dummy", "video": str(video)}) + "\n", encoding="utf-8")

    refs = Path("data/dev-mini/captioning/refs.jsonl")
    outdir = tmp_path / "out"

    code = run([
        "--manifest",
        str(manifest),
        "--engine",
        "fusion",
        "--output-dir",
        str(outdir),
        "--refs",
        str(refs),
    ])
    assert code == 0
    assert (outdir / "preds.jsonl").exists()
    assert (outdir / "summary.json").exists()

