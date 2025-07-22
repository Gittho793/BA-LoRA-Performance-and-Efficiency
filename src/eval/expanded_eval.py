
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
# from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from vllm import LLM, SamplingParams
from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel)
from deepeval_eval import integrate_deepeval_metrics, evaluate_with_deepeval


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
    parser.add_argument('--use_questions', action='store_true',
                        help='Whether to use the extracted questions')

    # Question extraction arguments
    parser.add_argument('--extract-questions', action='store_true',
                        help='Extract questions from ground truth files')
    parser.add_argument('--question-model', default='unsloth/Meta-Llama-3.1-8B-Instruct',
                        help='Model for question extraction (local vLLM)')
    parser.add_argument('--question-extraction-prompt',
                        default="""Generate 3-5 relevant questions that can be answered based on the following text. Return only the questions, one per line, numbered 1-5.

Text: {text}

Questions:
1.""",
                        help='Prompt template for question extraction')
    parser.add_argument('--max-questions', type=int, default=5,
                        help='Maximum number of questions to extract per text')

    # File paths
    parser.add_argument('--ground-truth', required=True,
                        help='Folder of ground truth .txt files')
    parser.add_argument('--predictions', default='output/predictions',
                        help='Folder to save or load predictions')
    parser.add_argument('--extracted-questions', default='output/questions',
                        help='Folder to save extracted questions')

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


def read_text_files(folder):
    """Read all text files from a folder"""
    texts = {}
    for fname in os.listdir(folder):
        if fname.endswith('.txt'):
            with open(os.path.join(folder, fname), 'r', encoding='utf-8') as f:
                texts[fname] = f.read().strip()
    return texts


def save_predictions(preds, out_folder):
    """Save predictions to files"""
    os.makedirs(out_folder, exist_ok=True)
    for fname, text in preds.items():
        with open(os.path.join(out_folder, fname), 'w', encoding='utf-8') as f:
            f.write(text)


def save_questions(questions, out_folder):
    """Save extracted questions to files"""
    os.makedirs(out_folder, exist_ok=True)
    for fname, question_list in questions.items():
        question_fname = fname.replace('.txt', '_questions.txt')
        with open(os.path.join(out_folder, question_fname), 'w', encoding='utf-8') as f:
            for i, question in enumerate(question_list, 1):
                f.write(f"{i}. {question.strip()}\n")


def extract_questions_with_vllm(texts, args):
    """Extract questions from text using vLLM with simple text output"""
    logger.info(f"Loading vLLM model: {args.question_model}")

    llm = LLM(
        model=args.question_model,
        max_model_len=3072,
        gpu_memory_utilization=0.6,
        tensor_parallel_size=1
    )

    # Simple prompt - no JSON complications
    simple_prompt = """Generiere 3-5 relevante inhaltliche Fragen, die auf der Grundlage des folgenden Textes beantwortet werden können.
Gebe nur die Fragen zurück, eine pro Zeile, nummeriert von 1-5. Die Fragen sollten inhaltlich sein.

Text: {text}

Fragen:
1."""

    # Improved sampling parameters
    sampling_params = SamplingParams(
        temperature=0.3,  # More deterministic
        top_p=0.9,
        max_tokens=800,   # Increased capacity
        stop=["\n\n", "Text:", "\n6."]  # Better stop conditions
    )

    extracted_questions = {}

    for fname, text in tqdm(texts.items(), desc="Extracting questions"):
        try:
            prompt = simple_prompt.format(text=text[:2000])
            outputs = llm.generate([prompt], sampling_params)
            generated_text = "1." + outputs[0].outputs[0].text

            # Direct text parsing - no JSON issues
            questions = parse_questions_from_text(
                generated_text, args.max_questions)

            extracted_questions[fname] = questions
            logger.info(f"Extracted {len(questions)} questions from {fname}")

        except Exception as e:
            logger.error(f"Error extracting questions from {fname}: {str(e)}")
            extracted_questions[fname] = []
    del llm
    cleanup()
    return extracted_questions


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
    for fname, prompt in tqdm(inputs.items(), desc="Generating predictions"):
        try:
            """input_ids = tokenizer(
                prompt, return_tensors='pt', truncation=True, max_length=2048, return_attention_mask=True
            ).input_ids.to(args.device)"""
            encoding = tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=2048,
                padding='longest',
                return_attention_mask=True,
            )
            input_ids = encoding.input_ids.to(args.device)
            attention_mask = encoding.attention_mask.to(args.device)

            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    attention_mask = attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Only decode the newly generated tokens
            new_tokens = out[0][len(input_ids[0]):]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            preds[fname.replace('.txt', '_pred.txt')] = text

        except Exception as e:
            logger.error(f"Error generating prediction for {fname}: {str(e)}")
            preds[fname.replace('.txt', '_pred.txt')] = ""

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
        torch.cuda.set_per_process_memory_fraction(0.7)  # Use 70% of GPU memory max

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
    chunk_size = 2  # Process 5 files at a time
    file_items = list(gt_texts.items())

    for chunk_start in range(0, len(file_items), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(file_items))
        chunk_items = file_items[chunk_start:chunk_end]

        # Clear GPU cache before processing each chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for fname, gt in chunk_items:
            pred_fname = fname.replace('.txt', '_pred.txt')
            pred = pred_texts.get(pred_fname, '')

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
                        print(f"GPU OOM for {fname}, falling back to smaller model...")
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


def main():
    try:
        args = parse_args()
        logger.info(f"Arguments: {args}")

        # Read ground truth texts
        gt_texts = read_text_files(args.ground_truth)
        logger.info(f"Loaded {len(gt_texts)} ground truth files")

        # Question extraction phase
        if args.extract_questions:
            logger.info("Starting question extraction...")
            extracted_questions = extract_questions_with_vllm(gt_texts, args)
            save_questions(extracted_questions, args.extracted_questions)

            # Evaluate question quality
            question_results = evaluate_questions(
                extracted_questions, gt_texts, args)

            # Print question extraction results
            print("\n=== Question Extraction Results ===")
            for fname, metrics in question_results.items():
                print(f"File: {fname}")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
                print()

            # Save question extraction results
            with open(f"../../results/question_extraction_results.json", 'w') as f:
                json.dump(question_results, f, indent=2)

        questions = {}

        if args.generate:
            logger.info(f"Loading model: {args.model_name}")

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model_name,
                load_in_4bit=False
            )

            FastLanguageModel.for_inference(model)

            if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
                # Choose a pad token that does not conflict
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # Update model embeddings if you added a new token
                model.resize_token_embeddings(len(tokenizer))


            # Use extracted questions as prompts, or original texts
            if args.use_questions and os.path.exists(args.extracted_questions):
                logger.info("Using extracted questions as prompts")
                question_files = read_text_files(args.extracted_questions)

                # Convert questions to prompts
                prompts = {}
                for fname, questions in question_files.items():
                    original_fname = fname.replace('_questions.txt', '.txt')
                    gt_text = gt_texts.get(original_fname, "")

                    # Create prompt combining context and questions
                    prompt = f"Context: {gt_text[:1000]}\n\nQuestions: {questions}\n\nAnswers:"
                    prompts[original_fname] = prompt
                    print(questions)
            else:
                prompts = gt_texts

            # Generate predictions
            preds = generate_predictions(model, tokenizer, prompts, args)
            save_predictions(preds, args.predictions)
        else:
            # Load existing predictions
            preds = read_text_files(args.predictions)

        if args.deepeval:
            logger.info("Running DeepEval evaluation...")
            deepeval_results = evaluate_with_deepeval(gt_texts, preds, questions)

            print("\n=== DeepEval Results ===")
            for fname, metrics in deepeval_results.items():
                print(f"File: {fname}")
                for metric, details in metrics.items():
                    print(f"  {metric}: {details['score']:.4f}, Success: {details['success']}, Reason: {details['reason']}")

            with open(f"../../results/deepeval_results.json", 'w', encoding="utf-8") as f:
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
                output = {"per_file": results, "aggregated": all_scores }
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
