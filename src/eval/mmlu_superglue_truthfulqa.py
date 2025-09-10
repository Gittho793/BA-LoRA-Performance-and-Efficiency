"""
Comprehensive script for evaluating LLMs using Unsloth with MMLU and SuperGLUE benchmarks.

This script demonstrates:
1. Loading a fine-tuned model with Unsloth and LoRA adapters
2. Evaluating on MMLU benchmark using LM Evaluation Harness
3. Evaluating on SuperGLUE benchmark 
"""

import os
import re
import json
import subprocess
import argparse
from typing import Dict, List, Optional
import datetime
import logging
from time import time
import torch
from src.util.args import MODEL_NAME

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse CLI arguments
    """
    parser = argparse.ArgumentParser(
        description="Offline LLM Evaluation for mmlu, superglue, and truthfulqa"
    )

    parser.add_argument("--adapter-path", default=None)

    return parser.parse_args()


class UnslothEvaluator:
    """Main class for evaluating Unsloth fine-tuned models"""

    def __init__(self,
                 model_name: str,
                 adapter_path: Optional[str] = None,
                 max_seq_length: int = 2048,
                 load_in_4bit: bool = True,
                 use_cache: bool = True):
        """
        Initialize the evaluator

        Args:
            model_name: Base model name (e.g., "unsloth/Meta-Llama-3.1-8B-Instruct")
            adapter_path: Path to LoRA adapter if using fine-tuned model
            max_seq_length: Maximum sequence length
            load_in_4bit: Whether to load model in 4-bit quantization
            use_cache: Whether to use caching for evaluation
        """
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.use_cache = use_cache

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.results = {}

    def load_model(self) -> bool:
        """Load the model with Unsloth optimizations"""
        try:
            from unsloth import FastLanguageModel

            logger.info(f"Loading model: {self.model_name}")
            if self.adapter_path:
                logger.info(f"Loading LoRA adapter from: {self.adapter_path}")

            # Load model and tokenizer
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.adapter_path if self.adapter_path else self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=self.load_in_4bit,
            )

            # Enable inference mode for faster evaluation
            FastLanguageModel.for_inference(self.model)

            # compile model for even faster evaluation
            self.model = torch.compile(self.model)

            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def evaluate_mmlu(self,
                      num_fewshot: int = 5,
                      limit: Optional[int] = None,
                      batch_size: str = "auto") -> Dict:
        """
        Evaluate model on MMLU benchmark using LM Evaluation Harness

        Args:
            num_fewshot: Number of few-shot examples (typically 5 for MMLU)
            limit: Limit number of examples for testing (None for full evaluation)
            batch_size: Batch size for evaluation ("auto" for automatic)

        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting MMLU evaluation...")

        # Prepare model arguments for lm_eval
        if self.adapter_path:
            model_args = f"pretrained={self.model_name},peft={self.adapter_path}"
        else:
            model_args = f"pretrained={self.model_name}"

        if self.load_in_4bit:
            model_args += ",load_in_4bit=True"

        sanitized_model_name = self.model_name.replace("/", "_")

        # Build lm_eval command
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", model_args,
            "--tasks", "mmlu",
            "--num_fewshot", str(num_fewshot),
            "--batch_size", batch_size,
            "--device", "cuda" if torch.cuda.is_available() else "cpu",
            "--output_path", f"../../results/mmlu/{sanitized_model_name if not self.adapter_path else os.path.basename(self.adapter_path)}/mmlu_{sanitized_model_name if not self.adapter_path else os.path.basename(self.adapter_path)}.json",
            # "--log_samples"
        ]

        if limit:
            cmd.extend(["--limit", str(limit)])

        if self.use_cache:
            cmd.extend(["--use_cache", "./cache"])

        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True)

            logger.info("MMLU evaluation completed successfully")
            self.results['mmlu'] = {
                'command': ' '.join(cmd),
                'stdout': result.stdout,
                # 'stderr': result.stderr
            }

            return self._parse_results(result.stdout, 'mmlu')

        except subprocess.CalledProcessError as e:
            logger.error(f"MMLU evaluation failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            return {}

    def evaluate_superglue(self,
                           num_fewshot: int = 0,
                           limit: Optional[int] = None,
                           batch_size: str = "auto") -> Dict:
        """
        Evaluate model on SuperGLUE benchmark

        Args:
            num_fewshot: Number of few-shot examples (typically 0-5 for SuperGLUE tasks)
            limit: Limit number of examples for testing
            batch_size: Batch size for evaluation

        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting SuperGLUE evaluation...")

        # SuperGLUE tasks
        superglue_tasks = [
            "boolq", "cb", "copa", "multirc", "record",
            "rte", "wic", "wsc"
        ]

        results = {}

        for task in superglue_tasks:
            logger.info(f"Evaluating {task}...")

            # Prepare model arguments
            if self.adapter_path:
                model_args = f"pretrained={self.model_name},peft={self.adapter_path}"
            else:
                model_args = f"pretrained={self.model_name}"

            if self.load_in_4bit:
                model_args += ",load_in_4bit=True"

            sanitized_model_name = self.model_name.replace("/", "_")

            cmd = [
                "lm_eval",
                "--model", "hf",
                "--model_args", model_args,
                "--tasks", task,
                "--num_fewshot", str(num_fewshot),
                "--batch_size", batch_size,
                "--device", "cuda" if torch.cuda.is_available() else "cpu",
                "--output_path", f"../../results/superglue/{sanitized_model_name if not self.adapter_path else os.path.basename(self.adapter_path)}/{task}.json",
                # "--log_samples"
            ]

            if limit:
                cmd.extend(["--limit", str(limit)])

            if self.use_cache:
                cmd.extend(["--use_cache", "./cache"])

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=True)

                results[task] = self._parse_results(result.stdout, task)
                logger.info(f"{task} evaluation completed")

            except subprocess.CalledProcessError as e:
                logger.error(f"{task} evaluation failed: {e}")
                results[task] = {}

        self.results['superglue'] = results
        return results

    def evaluate_truthfulqa(self,
                            num_fewshot: int = 5,
                            limit: Optional[int] = None,
                            batch_size: str = "auto",
                            tasks: Optional[List[str]] = None) -> Dict:
        """
        Evaluate model on TruthfulQA benchmark

        Args:
            num_fewshot: Number of few-shot examples (typically 0 for TruthfulQA)
            limit: Limit number of examples for testing
            batch_size: Batch size for evaluation
            tasks: Specific TruthfulQA tasks to run (None for all)

        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting TruthfulQA evaluation...")

        # TruthfulQA tasks available in lm-eval-harness
        if tasks is None:
            truthfulqa_tasks = [
                "truthful_qa_mc1",  # Multiple choice (single correct answer)
                # Multiple choice (multiple correct answers)
                "truthful_qa_mc2",
                "truthful_qa_gen"   # Generative (requires additional setup)
            ]
        else:
            truthfulqa_tasks = tasks

        results = {}

        for task in truthfulqa_tasks:
            logger.info(f"Evaluating {task}...")

            # Prepare model arguments
            if self.adapter_path:
                model_args = f"pretrained={self.model_name},peft={self.adapter_path}"
            else:
                model_args = f"pretrained={self.model_name}"

            if self.load_in_4bit:
                model_args += ",load_in_4bit=True"

            sanitized_model_name = self.model_name.replace("/", "_")

            cmd = [
                "lm_eval",
                "--model", "hf",
                "--model_args", model_args,
                "--tasks", task,
                "--num_fewshot", str(num_fewshot),
                "--batch_size", batch_size,
                "--device", "cuda" if torch.cuda.is_available() else "cpu",
                "--output_path", f"../../results/truthfulqa/{sanitized_model_name if not self.adapter_path else os.path.basename(self.adapter_path)}/{task}.json",
                # "--log_samples"
            ]

            if limit:
                cmd.extend(["--limit", str(limit)])

            if self.use_cache:
                cmd.extend(["--use_cache", "./cache"])

            try:
                logger.info(f"Running command: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=True)
                results[task] = self._parse_results(result.stdout, task)
                logger.info(f"{task} evaluation completed")

            except subprocess.CalledProcessError as e:
                logger.error(f"{task} evaluation failed: {e}")
                logger.error(f"stdout: {e.stdout}")
                logger.error(f"stderr: {e.stderr}")
                results[task] = {}

        self.results['truthfulqa'] = results
        return results

    def _parse_results(self, output: str, benchmark: str) -> Dict:
        lines = output.split('\n')
        results = {}
        in_results = False

        for line in lines:
            line = line.strip()

            # Use re for robustness
            if re.match(r'^\|\s*Tasks\b.*\bMetric\b.*\bValue\b', line):
                in_results = True
                continue

            # Skip divider lines
            if in_results and line.startswith("|-----"):
                continue

            # Parse actual result rows
            if in_results and line.startswith("|") and line.count("|") >= 9:
                parts = [p.strip() for p in line.split("|")]
                try:
                    # Remove first and last empty elements from split
                    if len(parts) > 0 and parts[0] == '':
                        parts = parts[1:]
                    if len(parts) > 0 and parts[-1] == '':
                        parts = parts[:-1]

                    if len(parts) >= 7:  # Ensure enough columns
                        task_name = parts[0] or benchmark
                        metric = parts[4]  # Metric column
                        value_str = parts[6]  # Value column

                        if value_str.upper() == "N/A":
                            logger.warning(
                                f"Skipping line due to N/A value: {line}")
                            continue

                        value = float(value_str)
                        results[f"{task_name}_{metric}"] = value

                except (IndexError, ValueError) as e:
                    logger.warning(f"Could not parse line: {line} — {e}")
                    continue

        return results

    def save_results(self, filepath: str | None = None):
        """Save evaluation results to file"""
        if not filepath:
            if self.adapter_path:
                # Use the last subdirectory from the adapter_path
                last_dir = os.path.basename(
                    os.path.normpath(self.adapter_path))
                filepath = f"../../results/mmlu_superglue_truthfulqa/mmlu_suglue_tqa_{last_dir}.json"
            else:
                # Use model name as fallback
                sanitized_model_name = self.model_name.replace("/", "_")
                filepath = f"../../results/mmlu_superglue_truthfulqa/mmlu_suglue_tqa_{sanitized_model_name}.json"
        else:
            sanitized_model_name = self.model_name.replace("/", "_")
            filepath = f"../../results/mmlu_superglue_truthfulqa/mmlu_suglue_tqa_{sanitized_model_name}.json"

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Results saved to {filepath}")


def main():
    """Main execution function"""

    args = parse_args()

    # Initialize evaluator
    evaluator = UnslothEvaluator(
        model_name=MODEL_NAME,
        adapter_path=args.adapter_path,  # None for base model evaluation
        max_seq_length=2048,
        load_in_4bit=True,  # for faster evals
        use_cache=False  # caching causes same results for different adapters
    )

    if not evaluator.load_model():
        logger.error("Failed to load model. Exiting.")
        return

    start = time()

    logger.info("=== MMLU Evaluation ===")
    mmlu_results = evaluator.evaluate_mmlu(num_fewshot=4,  # https://arxiv.org/pdf/2502.14502 uses 4
                                           limit=None)
    print("MMLU Results:", mmlu_results)

    logger.info("=== SuperGLUE Evaluation ===")
    superglue_results = evaluator.evaluate_superglue(num_fewshot=4,  # https://arxiv.org/pdf/2502.14502 uses 4
                                                     limit=None)
    print("SuperGLUE Results:", superglue_results)

    logger.info("=== TruthfulQA Evaluation ===")
    truthfulqa_results = evaluator.evaluate_truthfulqa(
        num_fewshot=4,  # https://arxiv.org/pdf/2502.14502 uses 4
        limit=None,
        tasks=["truthfulqa_mc1", "truthfulqa_mc2"]  # Skip generative
    )
    print("TruthfulQA Results:", truthfulqa_results)

    evaluator.save_results(
        filepath=evaluator.model_name if not evaluator.adapter_path else None)
    elapsed = time() - start
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Elapsed time: ", elapsed)

    logger.info("Evaluation completed successfully!")


def setup_environment():
    """Set up the evaluation environment"""
    # Create necessary directories
    os.makedirs("cache", exist_ok=True)

    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For debugging CUDA issues
    os.environ["NUMEXPR_MAX_THREADS"] = "16"  # Adjust based on slurm job


def get_available_tasks():
    """Get list of available evaluation tasks"""
    try:
        result = subprocess.run(["lm_eval", "--tasks", "list"],
                                capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get available tasks: {e}")
        return ""


if __name__ == "__main__":
    # Setup environment
    setup_environment()

    main()
