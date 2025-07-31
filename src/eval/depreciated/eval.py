'''
Offline LLM Evaluation Pipeline using DeepEval

This script reads ground-truth text files, prompts an LLM for predictions,
and computes evaluation metrics using the deepeval library.
'''
import unsloth
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from unsloth import FastLanguageModel
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline LLM Evaluation without YAML config"
    )
    parser.add_argument('--model-name', required=True,
                        help='Hugging Face model path or repo ID')
    parser.add_argument('--device', default='cpu',
                        help='Device for inference (cpu or cuda)')
    parser.add_argument('--generate', action='store_true',
                        help='Whether to run model.generate()')
    parser.add_argument('--ground-truth', default='../../data/raw_splitted_txt',
                        help='Folder of ground truth .txt files')
    parser.add_argument('--predictions', default='output/predictions',
                        help='Folder to save or load predictions')
    parser.add_argument('--max-new-tokens', type=int, default=4096,
                        help='Max new tokens for generation')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--do-sample', action='store_true',
                        help='Use sampling instead of greedy decode')
    parser.add_argument('--bleu', action='store_true', help='Compute BLEU')
    parser.add_argument('--bleu-type', choices=["all", 'bleu1', 'bleu2', 'bleu3', 'bleu4'],
                        default='all', help='BLEU type')
    parser.add_argument('--rouge', action='store_true', help='Compute ROUGE')
    parser.add_argument('--bert-score', action='store_true',
                        help='Compute BERTScore')
    parser.add_argument('--bert-model', default='google-bert/bert-base-multilingual-cased',
                        help='Model for BERTScore')
    parser.add_argument('--bert-lang', default='de',
                        help='Language for BERTScore')
    return parser.parse_args()


def read_text_files(folder):
    texts = {}
    for fname in os.listdir(folder):
        if fname.endswith('.txt'):
            with open(os.path.join(folder, fname), 'r', encoding='utf-8') as f:
                texts[fname] = f.read().strip()
    return texts


def save_predictions(preds, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    for fname, text in preds.items():
        with open(os.path.join(out_folder, fname), 'w', encoding='utf-8') as f:
            f.write(text)


"""def generate_predictions(model, tokenizer, inputs, args):
    preds = {}
    for fname, prompt in tqdm(inputs.items(), desc="Generating"):
        input_ids = tokenizer(
            prompt, return_tensors='pt').input_ids.to(args.device)
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        preds[fname.replace('.txt', '_pred.txt')] = text
    return preds"""


def generate_predictions(model, tokenizer, inputs, args):
    preds = {}
    for fname, prompt in tqdm(inputs.items(), desc="Generating"):
        input_ids = tokenizer(
            prompt, return_tensors='pt').input_ids.to(args.device)
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature
        )

        new_tokens = out[0][len(input_ids[0]):]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        preds[fname.replace('.txt', '_pred.txt')] = text
    return preds


def evaluate(gt_texts, pred_texts, args):
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
            precision, recall, f1_score = bert_score(
                [pred], [gt], 
                model_type=args.bert_model,
                lang=args.bert_lang, 
                rescale_with_baseline=True
            )
            res['bert_precision'], res['bert_recall'], res['bert_f1'] = \
                precision[0].item(), recall[0].item(), f1_score[0].item()
        results[fname] = res
    return results


def main():
    args = parse_args()

    print(args)
    gt_texts = read_text_files(args.ground_truth)
    """tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name).to(args.device)"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        load_in_4bit=False
    )

    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"

    if args.generate:
        preds = generate_predictions(model, tokenizer, gt_texts, args)
        save_predictions(preds, args.predictions)
    else:
        preds = read_text_files(args.predictions)

    results = evaluate(gt_texts, preds, args)
    for fname, metrics in results.items():
        print(f"Results for {fname}:")
        for m, v in metrics.items():
            print(f"\t{m}: {v:.4f}")

    # aggregate statistics
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

    # save stats
    just_model_name = os.path.basename(args.model_name)
    with open(f"../../results/{just_model_name}", 'w', encoding="utf-8") as f:
        output = {"per_file": results, "aggregated": all_scores}
        json.dump(output, f, indent=2)


if __name__ == '__main__':
    main()
