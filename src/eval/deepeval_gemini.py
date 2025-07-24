from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
    GEval,
)
from deepeval.test_case.llm_test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import GeminiModel  
import os
import signal
import contextlib
import sys

# ──────────────────────────────────────────────
# 1.  Build a single Gemini judge to reuse
# ──────────────────────────────────────────────
EVAL_MODEL = GeminiModel(
    model_name="gemini-2.0-flash-lite",          # or any free-tier model e.g. "gemini-1.5-pro"
    api_key=os.getenv("GEMINI_API_KEY"),  # picked up automatically if set
    temperature=0,
)

# ──────────────────────────────────────────────
# 2.  Metric factory
# ──────────────────────────────────────────────
def integrate_deepeval_metrics():
    """Return a dict of DeepEval metrics that all use Gemini."""
    return {
        "answer_relevancy": AnswerRelevancyMetric(
            threshold=0.7, model=EVAL_MODEL
        ),
        "faithfulness": FaithfulnessMetric(
            threshold=0.7, model=EVAL_MODEL
        ),
        "contextual_relevancy": ContextualRelevancyMetric(
            threshold=0.7, model=EVAL_MODEL
        ),
        "hallucination": HallucinationMetric(
            threshold=0.3, model=EVAL_MODEL
        ),
        "factual_correctness": GEval(
            name="Factual Correctness",
            criteria="Determine whether the actual output is factually correct based on the given context",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT],
            model=EVAL_MODEL,
        ),
    }

# ──────────────────────────────────────────────
# 3.  Utility helpers (unchanged except imports)
# ──────────────────────────────────────────────
def parse_questions_string(questions_string):
    if not questions_string:
        return {}
    questions_dict, current = {}, []
    for line in questions_string.strip().split("\n"):
        if line and line[0].isdigit() and ". " in line:
            current.append(line.split(". ", 1)[1])
    if current:
        questions_dict["default"] = current
    return questions_dict


def evaluate_with_deepeval(gt_texts, pred_texts, questions=None):
    """Run evaluation; returns nested dict of scores."""
    metrics = integrate_deepeval_metrics()
    results = {}

    # graceful-interrupt handlers (retain if desired)
    def _handler(signum, frame):
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")
    original_sigint = signal.signal(signal.SIGINT, _handler)
    original_sigterm = signal.signal(signal.SIGTERM, _handler)

    if isinstance(questions, str):
        questions_dict = parse_questions_string(questions)
    elif isinstance(questions, dict) or questions is None:
        questions_dict = questions or {}
    else:
        raise ValueError("questions must be str, dict or None")

    try:
        for fname, gt_text in gt_texts.items():
            pred_fname = fname.replace(".txt", "_pred.txt")
            pred_text = pred_texts.get(pred_fname, "")
            if not pred_text:
                continue

            q_list = questions_dict.get(fname, questions_dict.get("default", []))  # inputtext wrong
            q_list = q_list.split('\n') if q_list else []
            input_text = '\n'.join(q_list) if q_list else "Generate response based on context"

            tc = LLMTestCase(input=input_text,
                             actual_output=pred_text,
                             context=[gt_text],
                             retrieval_context=[gt_text])

            file_results = {}
            for mname, metric in metrics.items():
                try:
                    metric.measure(tc)
                    file_results[mname] = {
                        "score": metric.score,
                        "success": metric.success,
                        "reason": getattr(metric, "reason", "N/A"),
                    }
                except Exception as e:
                    file_results[mname] = {
                        "score": 0.0,
                        "success": False,
                        "reason": f"Error: {e}",
                    }

            results[fname] = file_results
            print(f"Evaluated {fname}: {len(file_results)} metrics")
        return results

    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
