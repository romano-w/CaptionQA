from pathlib import Path
import json

def test_vqa_baseline_with_manifest_and_refs(monkeypatch, tmp_path: Path):
    # Stub the Qwen VQA engine to avoid heavy deps
    import captionqa.qa.engines.qwen_vl_qa as qmod

    class Stub:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def from_configs(cls, *args, **kwargs):
            return cls()

        def answer(self, video_path: str, question: str, *, context=None):
            return "black"  # align with dev-mini refs

    monkeypatch.setattr(qmod, "QwenVLVQAEngine", Stub)

    from captionqa.qa.baseline_vqa import run

    # Manifest with the dev-mini dummy video and a simple question
    video = Path("data/dev-mini/samples/dummy.mp4")
    assert video.exists()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": "qa1", "video": str(video), "question": "What color is the scene?"}) + "\n", encoding="utf-8")

    refs = Path("data/dev-mini/qa/refs.jsonl")
    outdir = tmp_path / "out"
    code = run(["--manifest", str(manifest), "--refs", str(refs), "--output-dir", str(outdir)])
    assert code == 0
    assert (outdir / "preds.jsonl").exists()
    assert (outdir / "summary.json").exists()

