from captionqa.qa.eval import EvaluationResult
from captionqa.qa.summary import summarize_results


def test_summarize_results_accuracy_f1():
    results = [
        EvaluationResult(question_id="1", question="q", prediction="yes", reference="yes"),
        EvaluationResult(question_id="2", question="q", prediction="no", reference="yes"),
        EvaluationResult(question_id="3", question="q", prediction="maybe", reference="maybe"),
    ]
    summary = summarize_results(results)
    assert "accuracy" in summary and "f1" in summary
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert 0.0 <= summary["f1"] <= 1.0

