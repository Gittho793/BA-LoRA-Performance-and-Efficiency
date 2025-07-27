
"""
Offline LLM Evaluation Pipeline with Question Extraction using DeepEval

This script reads ground-truth text files, extracts questions using an LLM,
prompts an LLM for predictions, and computes evaluation metrics using the deepeval library.

New features:
- Question extraction from ground truth text files using vLLM
- Support for local LLM inference with vLLM
- Configurable question extraction prompts
- Extended evaluation pipeline with question generation metrics
"""
import unsloth
import os
import json
import argparse
import re
import logging
import gc
import sys
import contextlib
import torch
import numpy as np
from tqdm import tqdm
from unsloth import FastLanguageModel
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel)
from src.eval.deepeval_openai import evaluate_with_deepeval


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline LLM Evaluation with Question Extraction"
    )

    # Model arguments
    parser.add_argument('--model-name', required=True,
                        help='Hugging Face model path or repo ID')
    parser.add_argument('--device', default='cpu',
                        help='Device for inference (cpu or cuda)')
    parser.add_argument('--generate', action='store_true',
                        help='Whether to run model.generate()')

    # File paths
    parser.add_argument('--ground-truth', required=True,
                        help='Folder of ground truth .txt files')
    parser.add_argument('--predictions', default='output/predictions',
                        help='Directory to save or load model-specific prediction files')
    parser.add_argument('--questions-json', required=True,
                        help='Path to questions and expected answer json')

    # Generation parameters
    parser.add_argument('--max-new-tokens', type=int, default=4096,
                        help='Max new tokens for generation')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--do-sample', action='store_true',
                        help='Use sampling instead of greedy decode')

    # Evaluation metrics
    parser.add_argument('--bleu', action='store_true', help='Compute BLEU')
    parser.add_argument('--bleu-type', choices=["all", 'bleu1', 'bleu2', 'bleu3', 'bleu4'],
                        default='all', help='BLEU type')
    parser.add_argument('--rouge', action='store_true', help='Compute ROUGE')
    parser.add_argument('--bert-score', action='store_true',
                        help='Compute BERTScore')
    parser.add_argument('--bert-model', default='distilbert-base-uncased',  # "google-bert/bert-base-german-cased",
                        help='Model for BERTScore')
    parser.add_argument('--bert-lang', default='de',
                        help='Language for BERTScore')
    parser.add_argument('--deepeval', action='store_true',
                        help='Evaluate using DeepEval metrics')

    return parser.parse_args()


def load_question_json(path):
    """Load questions and answers from JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    question_map, answer_map = {}, {}
    for item in data["questions"]:
        source_file = item["source_file"]  # e.g., "123.txt"
        question_map.setdefault(source_file, []).append(item["question"])
        answer_map.setdefault(source_file, []).append(item["answer"])

    return question_map, answer_map


def build_prompts(question_map):
    """
    Build a structured dict mapping from filename to a dict with:
    - intro: prompt intro text 
    - questions: list of numbered question dicts {"number": int, "text": str}
    - outro: prompt outro (e.g. "Antworten:")
    """
    prompts = {}
    for fname, questions in question_map.items():
        q_list = [{"number": i + 1, "text": q}
                  for i, q in enumerate(questions)]
        prompts[fname] = {
            "intro": "Bitte beantworte die folgenden Fragen:",
            "questions": q_list,
            "outro": "Antworten:"
        }

        # print human-readable prompt
        readable_prompt = f"\n\n{prompts[fname]['intro']}\n" + \
                          "\n".join(f"{q['number']}. {q['text']}" for q in q_list) + \
                          f"\n\n{prompts[fname]['outro']}"

    return prompts


def read_text_files(folder):
    """Read all text files from a folder"""
    texts = {}
    for fname in os.listdir(folder):
        if fname.endswith('.txt'):
            with open(os.path.join(folder, fname), 'r', encoding='utf-8') as f:
                texts[fname] = f.read().strip()
    return texts


def save_predictions(preds: dict, model_name: str, base_path: str = "output/predictions"):
    """Save predictions from the model with model name in filename"""
    just_model_name = os.path.basename(model_name)
    os.makedirs(base_path, exist_ok=True)
    pred_file = os.path.join(base_path, f"{just_model_name}_predictions.json")
    with open(pred_file, 'w', encoding='utf-8') as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
    return pred_file


def read_predictions(path: str):
    """Read predictions made by the model"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_questions(questions, out_folder):
    """Save extracted questions to files"""
    os.makedirs(out_folder, exist_ok=True)
    for fname, question_list in questions.items():
        question_fname = fname.replace('.txt', '_questions.txt')
        with open(os.path.join(out_folder, question_fname), 'w', encoding='utf-8') as f:
            for i, question in enumerate(question_list, 1):
                f.write(f"{i}. {question.strip()}\n")


def cleanup():
    """
    Based on https://github.com/vllm-project/vllm/issues/6544
    """
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()


def parse_questions_from_text(text, max_questions):
    """Parse questions from generated text"""
    questions = []

    # Split by newlines and look for question patterns
    lines = text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove numbering (1. 2. etc.)
        line = re.sub(r'^\d+\.\s*', '', line)

        # Check if it looks like a question
        if line.endswith('?') or any(wh in line.lower() for wh in ['what', 'when', 'where', 'why', 'who', 'how']):
            questions.append(line)

        if len(questions) >= max_questions:
            break

    return questions


def generate_predictions(model, tokenizer, inputs, args):
    """Generate predictions using the model"""
    preds = {}
    for fname, q_list in tqdm(inputs.items()):
        preds[fname] = []
        for q in q_list['questions']:
            prompt_text = f"Bitte beantworte die folgende Frage:\n{q['number']}. {q['text']}\nAntwort:"

            try:
                encoding = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_new_tokens,
                    padding="longest",
                    return_attention_mask=True,
                )
                input_ids = encoding.input_ids.to(args.device)
                attention_mask = encoding.attention_mask.to(args.device)

                with torch.no_grad():
                    out = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                new_tokens = out[0][len(input_ids[0]):]
                pred_text = tokenizer.decode(
                    new_tokens, skip_special_tokens=True)

            except Exception as e:
                logger.error(
                    f"Error generating prediction for question in {fname}: {str(e)}")
                pred_text = ""

            # Append dictionary with question, predicted answer
            preds[fname].append({
                "question": q['text'],  # just question text, without numbers/prefix
                "predicted_answer": pred_text,
            })

    return preds


def evaluate_questions(questions, ground_truth_texts, args):
    """Evaluate the quality of extracted questions"""
    results = {}

    for fname, question_list in questions.items():
        gt_text = ground_truth_texts.get(fname, "")

        # Simple metrics for question evaluation
        question_metrics = {
            'num_questions': len(question_list),
            'avg_question_length': np.mean([len(q.split()) for q in question_list]) if question_list else 0,
            'question_diversity': len(set(question_list)) / len(question_list) if question_list else 0,
        }

        # Check if questions are answerable from the ground truth
        answerable_count = 0
        for question in question_list:
            # Simple check: if question contains keywords from ground truth
            question_words = set(question.lower().split())
            gt_words = set(gt_text.lower().split())
            overlap = len(question_words & gt_words)
            if overlap > 2:  # Basic threshold
                answerable_count += 1

        question_metrics['answerability_ratio'] = answerable_count / \
            len(question_list) if question_list else 0
        results[fname] = question_metrics

    return results


def evaluate(gt_texts, pred_texts, args):
    """Evaluate predictions using various metrics"""

    rouge_inst = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    bleu_weights = {
        'bleu1': (1, 0, 0, 0),  # 100% weight on unigram
        'bleu2': (0.5, 0.5, 0, 0),  # 50% weight on each uni- and bigram
        'bleu3': (0.33, 0.33, 0.33, 0),  # 33.3% weight on uni-, bi- and 3-gram
        'bleu4': (0.25, 0.25, 0.25, 0.25)  # 25% on each n-gram (up to 4)
    }

    results = {}

    for fname, gt in gt_texts.items():
        pred_fname = fname.replace('.txt', '_pred.txt')
        pred = pred_texts.get(pred_fname, '')

        res = {}

        if args.bleu:
            if args.bleu_type == 'all':
                for btype, weights in bleu_weights.items():
                    res[btype] = sentence_bleu(
                        [gt.split()], pred.split(), weights=weights)
            else:
                weights = bleu_weights[args.bleu_type]
                res[args.bleu_type] = sentence_bleu(
                    [gt.split()], pred.split(), weights=weights)

        if args.rouge:
            sc = rouge_inst.score(gt, pred)
            res['rouge1'] = sc['rouge1'].fmeasure
            res['rouge2'] = sc['rouge2'].fmeasure
            res['rougeL'] = sc['rougeL'].fmeasure

        if args.bert_score:
            torch.cuda.empty_cache()
            precision, recall, f1_score = bert_score(
                [pred], [gt],
                model_type=args.bert_model,
                lang=args.bert_lang,
                batch_size=2,  # reduced from default 64 for vram
                rescale_with_baseline=False
            )
            res['bert_precision'], res['bert_recall'], res['bert_f1'] = \
                precision[0].item(), recall[0].item(), f1_score[0].item()

        results[fname] = res

    return results


def evaluate_optimized(gt_texts, pred_texts, args):
    """Optimized evaluate function with memory management for BERT score"""

    # Enable mixed precision if available
    use_amp = torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast')

    # Set memory fraction to prevent over-allocation
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(
            0.8)  # Use 80% of GPU memory max

    rouge_inst = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    bleu_weights = {
        'bleu1': (1, 0, 0, 0),
        'bleu2': (0.5, 0.5, 0, 0),
        'bleu3': (0.33, 0.33, 0.33, 0),
        'bleu4': (0.25, 0.25, 0.25, 0.25)
    }

    results = {}

    # Process in smaller chunks to manage memory
    chunk_size = 2  # Process 2 files at a time
    file_items = list(gt_texts.items())

    for chunk_start in range(0, len(file_items), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(file_items))
        chunk_items = file_items[chunk_start:chunk_end]

        # Clear GPU cache before processing each chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for fname, gt in chunk_items:
            pred = pred_texts.get(fname, '')

            res = {}

            # BLEU score calculation (no memory issues)
            if args.bleu:
                if args.bleu_type == 'all':
                    for btype, weights in bleu_weights.items():
                        res[btype] = sentence_bleu(
                            [gt.split()], pred.split(), weights=weights)
                else:
                    weights = bleu_weights[args.bleu_type]
                    res[args.bleu_type] = sentence_bleu(
                        [gt.split()], pred.split(), weights=weights)

            # ROUGE score calculation (no memory issues)
            if args.rouge:
                sc = rouge_inst.score(gt, pred)
                res['rouge1'] = sc['rouge1'].fmeasure
                res['rouge2'] = sc['rouge2'].fmeasure
                res['rougeL'] = sc['rougeL'].fmeasure

            # BERT score calculation with memory optimizations
            if args.bert_score:
                try:
                    with torch.amp.autocast("cuda", enabled=False):
                        precision, recall, f1_score = bert_score(
                            [pred], [gt],
                            model_type=args.bert_model,
                            lang=args.bert_lang,
                            batch_size=2,  # Minimum batch size
                            rescale_with_baseline=False,
                            device='cuda' if torch.cuda.is_available() else 'cpu'
                        )

                    res['bert_precision'] = precision[0].item()
                    res['bert_recall'] = recall[0].item()
                    res['bert_f1'] = f1_score[0].item()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(
                            f"GPU OOM for {fname}, falling back to smaller model...")
                        # Fallback to CPU processing
                        torch.cuda.empty_cache()
                        precision, recall, f1_score = bert_score(
                            [pred], [gt],
                            model_type='distilbert-base-uncased',
                            lang=args.bert_lang,
                            batch_size=1,
                            rescale_with_baseline=False,
                            device='cuda' if torch.cuda.is_available() else 'cpu'
                        )
                        res['bert_precision'] = precision[0].item()
                        res['bert_recall'] = recall[0].item()
                        res['bert_f1'] = f1_score[0].item()
                    else:
                        raise e

            results[fname] = res

            # Clean up variables
            del res
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Clean up chunk variables
        del chunk_items
        gc.collect()

    return results


def process_question_files(question_files, gt_texts):
    prompts = {}
    questions_dict = {}  # Collect all questions here
    for fname, q_text in question_files.items():  # Rename loop var to avoid shadowing
        original_fname = fname.replace('_questions.txt', '.txt')
        gt_text = gt_texts.get(original_fname, "")
        prompt = f"Context: {gt_text[:1000]}\n\nQuestions: {q_text}\n\nAnswers:"
        prompts[original_fname] = prompt
        # Store per-file questions string
        questions_dict[original_fname] = q_text
    return prompts, questions_dict


def main():
    try:
        args = parse_args()
        logger.info(f"Arguments: {args}")

        # Read ground truth texts
        gt_texts = read_text_files(args.ground_truth)
        logger.info(f"Loaded {len(gt_texts)} ground truth files")

        # Load questions and answers from JSON
        question_map, answer_map = load_question_json(args.questions_json)
        logger.info(f"Loaded questions for {len(question_map)} files")

        if args.generate:
            logger.info(f"Loading model: {args.model_name}")

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model_name,
                load_in_4bit=False
            )

            FastLanguageModel.for_inference(model)

            if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # Update model embeddings because added a new token
                model.resize_token_embeddings(len(tokenizer))

            prompts = build_prompts(question_map)

            # Generate predictions
            preds = generate_predictions(model, tokenizer, prompts, args)
            pred_file = save_predictions(
                preds, args.model_name, args.predictions)
            torch.cuda.empty_cache()
        else:
            # Load existing predictions
            just_model_name = os.path.basename(args.model_name)
            pred_file = os.path.join(
                args.predictions, f"{just_model_name}_predictions.json")
            preds = read_predictions(pred_file)

        if args.deepeval:
            logger.info("Running DeepEval evaluation...")

            deepeval_results = evaluate_with_deepeval(
                gt_texts, preds, question_map, answer_map)

            print("\n=== DeepEval Results ===")
            for fname, metrics in deepeval_results.items():
                print(f"File: {fname}")
                for metric, details in metrics.items():
                    print(
                        f"  {metric}: {details['score']:.4f}, Success: {details['success']}, Reason: {details['reason']}")
            just_model_name = os.path.basename(args.model_name)

            with open(f"../../results/deepeval_{just_model_name}_results.json", 'w', encoding="utf-8") as f:
                json.dump(deepeval_results, f, indent=2)

        # Evaluate predictions
        if any([args.bleu, args.rouge, args.bert_score]):
            logger.info("Evaluating predictions...")
            results = evaluate_optimized(gt_texts, preds, args)

            # Print results
            print("\n=== Prediction Evaluation Results ===")
            for fname, metrics in results.items():
                print(f"Results for {fname}:")
                for m, v in metrics.items():
                    print(f"\t{m}: {v:.4f}")

            # Aggregate statistics
            all_scores = {}
            for metrics_dict in results.values():
                for metric, score in metrics_dict.items():
                    if metric not in all_scores:
                        all_scores[metric] = []
                    all_scores[metric].append(score)

            print("\nAggregate Results:")
            for metric, scores in all_scores.items():
                mean_score = sum(scores) / len(scores)
                print(f"{metric}: {mean_score:.4f} ± {np.std(scores):.4f}")

            # Save results
            just_model_name = os.path.basename(args.model_name)
            os.makedirs("../../results", exist_ok=True)
            with open(f"../../results/{just_model_name}_results.json", 'w', encoding="utf-8") as f:
                output = {"per_file": results, "aggregated": all_scores}
                json.dump(output, f, indent=2)

    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise
    finally:
        torch.cuda.empty_cache()
        print("GPU cache cleared.")
        sys.exit()


if __name__ == '__main__':
    main()
