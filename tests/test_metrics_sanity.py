from captionqa.evaluation.metrics import compute_caption_metrics


def test_bleu_bounds_and_identity():
    preds = {"a": "the quick brown fox", "b": ""}
    refs = {
        "a": ["the quick brown fox"],
        "b": ["jumps over the lazy dog"],
    }
    metrics = compute_caption_metrics(preds, refs)
    bleu = metrics["bleu"].score
    assert 0.0 <= bleu <= 1.0
    # For identical sentence, expect near-perfect BLEU
    per = metrics["bleu"].metadata["per_example"][0]
    assert per["bleu"] >= 0.99
    # Precisions should be in [0,1]
    for p in per["precisions"]:
        assert 0.0 <= p <= 1.0


def test_spice_and_cider_nonnegative():
    preds = {"x": "a black scene with a tone"}
    refs = {"x": ["a black 360-degree scene with a short tone"]}
    metrics = compute_caption_metrics(preds, refs)
    assert metrics["cider"].score >= 0.0
    assert metrics["spice"].score >= 0.0

