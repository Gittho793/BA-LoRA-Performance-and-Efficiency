"""
Use the deepeval library to evaluate predictions of an LLM with ground truth text and expected answers
"""
from tqdm import tqdm
import os
import signal
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
    GEval,
)
from deepeval.test_case.llm_test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import GPTModel


EVAL_MODEL = GPTModel(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPEN_AI"),
    temperature=0,
)


def integrate_deepeval_metrics(include_contextual: bool = False):
    """Return a dict of DeepEval metrics, optionally including contextual relevancy (for RAG)."""
    metrics = {
        "answer_relevancy": AnswerRelevancyMetric(
            threshold=0.7, model=EVAL_MODEL
        ),
        "faithfulness": FaithfulnessMetric(
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
    if include_contextual:  # changed after evaluation finished but should work
        metrics["contextual_relevancy"] = ContextualRelevancyMetric(
            threshold=0.7, model=EVAL_MODEL
        )
    return metrics


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
                           retrieval_context = None
                           ) -> dict:
    """
    Evaluate every <question, predicted-answer, expected-answer> triple
    independently with DeepEval metrics.
    """
    metrics = integrate_deepeval_metrics(include_contextual=bool(retrieval_context))  # changed after evaluation finished but should work
    all_results = {}

    def _handler(signum, frame):
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")
    original_sigint = signal.signal(signal.SIGINT, _handler)
    original_sigterm = signal.signal(signal.SIGTERM, _handler)

    try:
        for fname, gt_text in tqdm(gt_texts.items()):
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
                if retrieval_context is None:
                    # Default: use ground truth text
                    context_for_retrieval = [gt_text]
                elif isinstance(retrieval_context, dict):
                    # Per-file, per-question contexts
                    file_contexts = retrieval_context.get(fname, [])
                    if i < len(file_contexts):
                        ctx = file_contexts[i]
                        context_for_retrieval = [ctx] if isinstance(ctx, str) else ctx
                    else:
                        context_for_retrieval = [gt_text]
                elif isinstance(retrieval_context, list):
                    # Single context list for all
                    context_for_retrieval = retrieval_context
                elif isinstance(retrieval_context, str):
                    # Single context string for all
                    context_for_retrieval = [retrieval_context]
                else:
                    # Fallback
                    context_for_retrieval = [gt_text]
                tc = LLMTestCase(
                    input=qs[i],
                    actual_output=predicted_answers[i],
                    expected_output=gold_as[i],
                    context=[gt_text],
                    retrieval_context=context_for_retrieval
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
