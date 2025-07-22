from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
    GEval
)
from deepeval.test_case import LLMTestCase


def integrate_deepeval_metrics():
    """Integration with DeepEval framework for comprehensive evaluation"""

    # Define metrics
    metrics = {
        'answer_relevancy': AnswerRelevancyMetric(threshold=0.7),
        'faithfulness': FaithfulnessMetric(threshold=0.7),
        'contextual_relevancy': ContextualRelevancyMetric(threshold=0.7),
        'hallucination': HallucinationMetric(threshold=0.3),
        'factual_correctness': GEval(
            name="Factual Correctness",
            criteria="Determine whether the actual output is factually correct based on the given context",
            evaluation_params=["input", "actual_output", "context"]
        )
    }

    return metrics


def evaluate_with_deepeval(gt_texts, pred_texts, questions=None):
    """Evaluate predictions using DeepEval metrics"""

    metrics = integrate_deepeval_metrics()
    results = {}

    for fname, gt_text in gt_texts.items():
        pred_fname = fname.replace('.txt', '_pred.txt')
        pred_text = pred_texts.get(pred_fname, '')

        if not pred_text:
            continue

        # Create test case
        test_case = LLMTestCase(
            input=questions.get(fname, "Generate response based on context"),
            actual_output=pred_text,
            context=[gt_text]  # DeepEval expects list for context
        )

        # Evaluate each metric
        file_results = {}
        for metric_name, metric in metrics.items():
            try:
                metric.measure(test_case)
                file_results[metric_name] = {
                    'score': metric.score,
                    'success': metric.success,
                    'reason': getattr(metric, 'reason', 'N/A')
                }
            except Exception as e:
                file_results[metric_name] = {
                    'score': 0.0,
                    'success': False,
                    'reason': f'Error: {e}'
                }

        results[fname] = file_results
    print("results", results)
    return results
