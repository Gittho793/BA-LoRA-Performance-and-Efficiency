
"""
Comprehensive script for evaluating LLMs using Unsloth with MMLU and SuperGLUE benchmarks.

This script demonstrates:
1. Loading a fine-tuned model with Unsloth and LoRA adapters
2. Evaluating on MMLU benchmark using LM Evaluation Harness
3. Evaluating on SuperGLUE benchmark 
4. Custom evaluation functions
5. Performance comparison between base and fine-tuned models

Requirements:
- unsloth
- lm_eval (EleutherAI LM Evaluation Harness)
- transformers
- torch
- datasets
- accelerate
"""

import os
import json
import torch
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from src.util.args import MODEL_NAME

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def install_requirements():
    """Install required packages"""
    packages = [
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "lm_eval[api] @ git+https://github.com/EleutherAI/lm-evaluation-harness.git",
        "--no-deps xformers<0.0.27 trl<0.9.0 peft accelerate bitsandbytes"
    ]

    for package in packages:
        logger.info(f"Installing {package}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + package.split())


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
            model_name: Base model name (e.g., "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
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

    def load_model(self):
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

            # Enable inference mode for 2x faster evaluation
            FastLanguageModel.for_inference(self.model)

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

        # Build lm_eval command
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", model_args,
            "--tasks", "mmlu",
            "--num_fewshot", str(num_fewshot),
            "--batch_size", batch_size,
            "--device", "cuda" if torch.cuda.is_available() else "cpu",
            "--output_path", f":./../mmlu_results/mmlu_{self.model_name if not self.adapter_path else self.model_name}",
            "--log_samples"
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
                'stderr': result.stderr
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

            cmd = [
                "lm_eval",
                "--model", "hf",
                "--model_args", model_args,
                "--tasks", task,
                "--num_fewshot", str(num_fewshot),
                "--batch_size", batch_size,
                "--device", "cuda" if torch.cuda.is_available() else "cpu",
                "--output_path", f"./results/{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "--log_samples"
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

    def _parse_results(self, output: str, benchmark: str) -> Dict:
        """Parse lm_eval output to extract scores"""
        lines = output.split('\n')
        results = {}

        # Look for result tables in output
        in_results = False
        for line in lines:
            line = line.strip()

            if 'Results' in line or 'Task' in line:
                in_results = True
                continue

            if in_results and '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 4:
                    task_name = parts[1] if parts[1] else benchmark
                    metric = parts[2] if len(parts) > 2 else 'score'
                    value = parts[3] if len(parts) > 3 else '0.0'

                    try:
                        value = float(value.replace('%', ''))
                        results[f"{task_name}_{metric}"] = value
                    except ValueError:
                        pass

        return results

    def save_results(self, filepath: str | None = None):
        """Save evaluation results to file"""
        if not filepath:
            if self.adapter_path:
                # Use the last subdirectory from the adapter_path
                last_dir = os.path.basename(os.path.normpath(self.adapter_path))
                filepath = f"../../results/mmlu_superglue/mmlu_suglue{last_dir}.json"
            else:
                # Use model name as fallback
                sanitized_model_name = self.model_name.replace("/", "_")
                filepath = f"../../results/mmlu_superglue/mmlu_suglue{sanitized_model_name}.json"

        with open(filepath, 'w',encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Results saved to {filepath}")


def main():
    """Main execution function with examples"""

    # Path to fine-tuned LoRA adapter
    adapter_path = "../../../unslothLora/Llama/txt-ds-new-llama-3.1-8b-v0.3-4bit-lora-r16-a16"

    # Initialize evaluator
    evaluator = UnslothEvaluator(
        model_name=MODEL_NAME,
        adapter_path=None,  # None for base model evaluation
        max_seq_length=2048,
        load_in_4bit=True,
        use_cache=True
    )

    # Load the model
    if not evaluator.load_model():
        logger.error("Failed to load model. Exiting.")
        return

    logger.info("=== MMLU Evaluation ===")
    mmlu_results = evaluator.evaluate_mmlu(num_fewshot=5, limit=100)
    print("MMLU Results:", mmlu_results)

    logger.info("=== SuperGLUE Evaluation ===")
    superglue_results = evaluator.evaluate_superglue(num_fewshot=0, limit=100)
    print("SuperGLUE Results:", superglue_results)

    evaluator.save_results(filepath=evaluator.model_name if not evaluator.adapter_path else None)

    logger.info("Evaluation completed successfully!")


def setup_environment():
    """Set up the evaluation environment"""
    # Create necessary directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("cache", exist_ok=True)

    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For debugging CUDA issues


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

    # Get available tasks
    tasks = get_available_tasks()
    if tasks:
        print("Available evaluation tasks:")
        print(tasks[:1000] + "..." if len(tasks) > 1000 else tasks)

    # Run main evaluation
    main()
