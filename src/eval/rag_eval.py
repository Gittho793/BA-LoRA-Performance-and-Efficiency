"""
Evaluate RAG pipeline results using the same metrics as expanded_eval.py
"""
from src.eval.deepeval_openai import evaluate_with_deepeval
from src.eval.expanded_eval import (
    load_question_json,
    read_text_files,
    evaluate_optimized_deepeval,
    aggregate_question_metrics,
    save_evaluation_results
)
import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def convert_rag_to_deepeval_format(rag_results, question_map, answer_map):
    """Convert RAG results to the same format as expanded_eval.py output"""
    deepeval_data = {}

    for filename, results_list in rag_results.items():
        if filename not in question_map or filename not in answer_map:
            continue

        deepeval_data[filename] = []

        questions = question_map[filename]
        expected_answers = answer_map[filename]

        for i, result in enumerate(results_list):
            if i < len(questions) and i < len(expected_answers):
                # Create the exact same structure as the target JSON
                deepeval_data[filename].append({
                    "question": questions[i],
                    "expected": expected_answers[i],
                    "predicted": result["predicted_answer"],
                    "existing_metrics": {},  # Empty dict like in target
                    # Don't include metrics here - they'll be added by evaluate_optimized_deepeval
                })

    return deepeval_data


def extract_retrieval_context(rag_results):
    """Extract retrieval context from RAG results in the format expected by deepeval_openai"""
    retrieval_context = {}

    for filename, results_list in rag_results.items():
        retrieval_context[filename] = []

        for result in results_list:
            # Extract the best_context for each question
            context = result.get("best_context", "")
            retrieval_context[filename].append(context)

    return retrieval_context


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG results")
    parser.add_argument('--rag-results', required=True,
                        help='Path to RAG results JSON')
    parser.add_argument('--questions-json', required=True,
                        help='Path to questions JSON')
    parser.add_argument('--ground-truth', required=True,
                        help='Path to ground truth files')
    parser.add_argument(
        '--output-name', default='rag_evaluation', help='Output file name prefix')
    parser.add_argument('--deepeval', action='store_true',
                        help='Run DeepEval metrics')
    parser.add_argument('--bleu', action='store_true', help='Compute BLEU')
    parser.add_argument('--bleu-type', choices=["all", 'bleu1', 'bleu2', 'bleu3', 'bleu4'],
                        default='all', help='BLEU type')
    parser.add_argument('--rouge', action='store_true', help='Compute ROUGE')
    parser.add_argument('--bert-score', action='store_true',
                        help='Compute BERTScore')

    args = parser.parse_args()

    # Load data
    with open(args.rag_results, 'r', encoding='utf-8') as f:
        rag_results = json.load(f)

    question_map, answer_map = load_question_json(args.questions_json)
    gt_texts = read_text_files(args.ground_truth)

    # Convert to prediction format for compatibility
    deepeval_data = convert_rag_to_deepeval_format(
        rag_results, question_map, answer_map)

    if args.deepeval:
        print("Running DeepEval evaluation...")

        preds = {}
        for filename, qa_list in deepeval_data.items():
            preds[filename] = [
                {
                    "question": qa["question"],
                    "predicted_answer": qa["predicted"]
                }
                for qa in qa_list
            ]

        retrieval_context = extract_retrieval_context(rag_results)

        """for filename, contexts in list(retrieval_context.items())[:2]:
            print(f"\nFile: {filename}")
            for i, ctx in enumerate(contexts[:2]):
                print(f"  Q{i+1} context preview: {ctx[:100]}...")"""

        deepeval_results = evaluate_with_deepeval(
            gt_texts, preds, question_map, answer_map,
            retrieval_context=retrieval_context)

        # Save DeepEval results
        os.makedirs("../../results/deepeval", exist_ok=True)
        with open(f"../../results/deepeval/deepeval_{args.output_name}_results.json", 'w', encoding='utf-8') as f:
            json.dump(deepeval_results, f, indent=2)

    # Run other metrics if specified
    if any([args.bleu, args.rouge, args.bert_score]):
        deepeval_data = convert_rag_to_deepeval_format(
            rag_results, question_map, answer_map)

        print("Evaluating with traditional metrics...")
        results = evaluate_optimized_deepeval(deepeval_data, args)
        aggregated = aggregate_question_metrics(results)

        # Save results
        os.makedirs("../../results/metrics_deepeval", exist_ok=True)
        save_evaluation_results(
            results,
            aggregated,
            f"../../results/metrics_deepeval/{args.output_name}_results.json"
        )

        print("\n=== Evaluation Summary ===")
        for metric, value in aggregated['overall'].items():
            if metric.endswith('_mean'):
                metric_name = metric.replace('_mean', '')
                std_key = metric.replace('_mean', '_std')
                count_key = metric.replace('_mean', '_count')
                std_value = aggregated['overall'].get(std_key, 0)
                count_value = aggregated['overall'].get(count_key, 0)
                print(
                    f"{metric_name}: {value:.4f} ± {std_value:.4f} (n={count_value})")


if __name__ == '__main__':
    main()
