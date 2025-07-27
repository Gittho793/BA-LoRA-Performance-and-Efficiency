from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
    GEval,
)
from deepeval.test_case.llm_test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import GPTModel
import os
import signal
import re


def _extract_numbered_answers(raw: str) -> list[str]:
    """
    Convert the model’s multiline string
       1. foo\n2. bar\n3. baz
    into ['foo', 'bar', 'baz'].

    Non-numbered lines are ignored; trailing whitespace is stripped.
    """
    answers = []
    for line in raw.splitlines():
        m = re.match(r"\s*\d+\.\s*(.*)", line)
        if m:
            answers.append(m.group(1).strip())
    # Fallback: treat the whole block as one answer
    if not answers and raw.strip():
        answers.append(raw.strip())
    return answers


# ──────────────────────────────────────────────
# 1.  Build a single Gemini judge to reuse
# ──────────────────────────────────────────────
EVAL_MODEL = GPTModel(
    model="gpt-4.1-mini",          # or any free-tier model e.g. "gemini-1.5-pro"
    api_key=os.getenv("OPEN_AI"),  # picked up automatically if set
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
            evaluation_params=[LLMTestCaseParams.INPUT,
                               LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT],
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


def evaluate_with_deepeval(gt_texts: dict,
                           pred_texts: dict,
                           question_map: dict,
                           answer_map: dict,
                           ) -> dict:
    """
    Evaluate every <question, predicted-answer, expected-answer> triple
    independently with DeepEval metrics.
    """
    metrics = integrate_deepeval_metrics()
    all_results = {}

    def _handler(signum, frame):
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")
    original_sigint = signal.signal(signal.SIGINT, _handler)
    original_sigterm = signal.signal(signal.SIGTERM, _handler)

    try:
        for fname, gt_text in gt_texts.items():
            preds_block = pred_texts.get(fname, [])
            predicted_answers = [entry['predicted_answer'] for entry in preds_block]

            qs = question_map.get(fname, [])
            gold_as = answer_map.get(fname, [])

            if not qs or not gold_as:
                # nothing to score for this file
                continue

            # Keep list lengths in sync (tolerate slight length mismatches)
            limit = min(len(qs), len(gold_as), len(predicted_answers))
            file_scores = []

            for i in range(limit):
                tc = LLMTestCase(
                    input=qs[i],
                    actual_output=predicted_answers[i],
                    expected_output=gold_as[i],
                    context=[gt_text],
                )

                q_score = {}
                for mname, metric in metrics.items():
                    try:
                        metric.measure(tc)
                        q_score[mname] = {
                            "score": metric.score,
                            "success": metric.success,
                            "reason": getattr(metric, "reason", "N/A"),
                        }
                    except Exception as e:
                        q_score[mname] = {
                            "score": 0.0,
                            "success": False,
                            "reason": f"Error: {e}",
                        }
                file_scores.append({
                    "question": qs[i],
                    "expected": gold_as[i],
                    "predicted": predicted_answers[i],
                    "metrics": q_score,
                })

            all_results[fname] = file_scores
            print(f"Evaluated {fname}: {len(file_scores)} Q-A pairs")

        return all_results

    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
