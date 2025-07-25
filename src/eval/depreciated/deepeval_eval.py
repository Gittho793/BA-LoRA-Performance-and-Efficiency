from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
    GEval
)
from deepeval.test_case.llm_test_case import LLMTestCase
import subprocess
import time
import os
import signal
import contextlib
import torch
import gc
import sys


def integrate_deepeval_metrics():
    """Integration with DeepEval framework for comprehensive evaluation"""

    # Define metrics
    metrics = {
        'answer_relevancy': AnswerRelevancyMetric(threshold=0.7, async_mode=False),
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


def parse_questions_string(questions_string):
    """Parse the concatenated questions string into individual questions"""
    if not questions_string:
        return {}

    questions_dict = {}
    lines = questions_string.strip().split('\n')
    current_file_questions = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if it's a numbered question
        if line[0].isdigit() and '. ' in line:
            question = line.split('. ', 1)[1] if '. ' in line else line
            current_file_questions.append(question)

    # Assign all questions to a default key
    if current_file_questions:
        questions_dict['default'] = current_file_questions

    return questions_dict


def cleanup_distributed_resources():
    """
    Enhanced cleanup function that properly destroys PyTorch distributed groups
    This should be called before process termination to avoid resource leaks
    """
    try:
        # First try to import and use vLLM's cleanup utilities
        from vllm.distributed import destroy_model_parallel, destroy_distributed_environment
        destroy_model_parallel()
        destroy_distributed_environment()
    except ImportError:
        print("vLLM distributed utilities not available, trying PyTorch cleanup")

    # Always try to destroy the default process group
    with contextlib.suppress(AssertionError, RuntimeError):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    # Force garbage collection and GPU memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Clear all CUDA contexts
        try:
            torch.cuda.synchronize()
        except:
            pass


def start_vllm_server(model_name, port=8000):
    """Deepeval needs external vllm server so its started here"""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--device", "cuda",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", "bfloat16",
        "--tensor-parallel-size", "1",
        "--max-num-batched-tokens", "2048",
        "--max-num-seqs", "16",
        "--cpu-offload-gb", "0",
    ]

    process = subprocess.Popen(
        cmd,
        start_new_session=True,  # Safer than preexec_fn
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    print(f"Starting vLLM server for model {model_name} on port {port}...")

    time.sleep(150)  # Give time for model loading and server startup

    return process


def terminate_process_tree(process):
    """
    Safely terminate a process and all its children
    Handles both Unix and Windows systems appropriately
    """
    if process.poll() is not None:
        return  # Process already terminated

    try:
        if hasattr(os, 'killpg'):  # Unix systems
            # First try gentle termination
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)

            # Wait a bit for graceful shutdown
            try:
                process.wait(timeout=30)
                print("vLLM server terminated gracefully")
                return
            except subprocess.TimeoutExpired:
                print("Graceful termination timed out, forcing shutdown...")

            # Force kill if graceful shutdown failed
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=10)

        else:  # Windows systems
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    except (OSError, ProcessLookupError) as e:
        print(
            f"Process cleanup completed (process may have already exited): {e}")
    except Exception as e:
        print(f"Error during process cleanup: {e}")


def evaluate_with_deepeval(gt_texts, pred_texts, model, questions=None):
    """Evaluate predictions using DeepEval metrics"""
    process = None

    # Set up signal handlers for cleanup
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, cleaning up...")
        if process:
            terminate_process_tree(process)
        cleanup_distributed_resources()
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")

    # Register signal handlers
    original_sigint = signal.signal(signal.SIGINT, signal_handler)
    original_sigterm = signal.signal(signal.SIGTERM, signal_handler)

    metrics = integrate_deepeval_metrics()
    results = {}

    # Parse questions if they're in string format
    if isinstance(questions, str):
        questions_dict = parse_questions_string(questions)
    elif isinstance(questions, dict):
        questions_dict = questions
    else:
        raise ValueError(
            "Something wrong with questions dict in eval w deepeval")

    try:

        process = start_vllm_server(model)

        for fname, gt_text in gt_texts.items():
            pred_fname = fname.replace('.txt', '_pred.txt')
            pred_text = pred_texts.get(pred_fname, '')

            if not pred_text:
                continue

            # Get the appropriate input question
            if questions_dict:
                file_questions = questions_dict.get(
                    fname, questions_dict.get('default', []))
                if file_questions:
                    # Use the first question or combine multiple questions
                    input_text = file_questions[0] if len(
                        file_questions) == 1 else " ".join(file_questions)
                else:
                    input_text = "Generate response based on context"
            else:
                input_text = "Generate response based on context"

            # Create test case with proper parameters
            test_case = LLMTestCase(
                input=input_text,
                actual_output=pred_text,
                context=[gt_text]  # DeepEval expects list for context
            )

            # Evaluate each metric with error handling
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
                        'reason': f'Error: {str(e)}'
                    }

            results[fname] = file_results
            print(f"Evaluated {fname}: {len(file_results)} metrics")

        return results

    except KeyboardInterrupt:
        print("Keyboard interrupt received! Cleaning up vLLM server and distributed resources...")
        raise

    finally:
        # Restore original signal handlers
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)

        # Clean up the vLLM server process
        if process:
            terminate_process_tree(process)

        # Clean up distributed resources
        cleanup_distributed_resources()

        print("Cleanup completed successfully.")
