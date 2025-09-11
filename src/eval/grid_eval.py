"""
Evaluate multiple models from one directory.
"""
import signal
import subprocess
import os
import sys
import datetime
from time import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("../../.env")  # necessary for local imports on cluster

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.util.args import (PDF_OUTPUT_DIR as OUTPUT_DIR,
                           PDF_GROUND_TRUTH_FILES as GROUND_TRUTH_FILES)


class ModelEvaluator:
    """Handles evaluation of multiple models in a directory."""
    
    def __init__(self, output_dir: str, ground_truth_files: str) -> None:
        self.output_dir = output_dir
        self.ground_truth_files = ground_truth_files
        self.base_dir = Path(output_dir).parent
        
    def _get_subdirectories_and_questions(self) -> tuple[List[Path], str]:
        """Get relevant subdirectories and questions file based on output type."""
        if "txt" in self.output_dir and "txt" in self.ground_truth_files:
            subdirs = [
                self.base_dir / name
                for name in sorted(os.listdir(self.base_dir))
                if name.startswith("txt") and (self.base_dir / name).is_dir()
            ]
            questions_json = "output/txt_questions.json"
        elif "pdf" in self.output_dir and "pdf" in self.ground_truth_files:
            subdirs = [
                self.base_dir / name
                for name in sorted(os.listdir(self.base_dir))
                if (name.startswith("pdf") and "4bit" in name and 
                    (self.base_dir / name).is_dir())
            ]
            questions_json = "output/pdf_questions.json"
        else:
            raise ValueError("GROUND TRUTH AND OUTPUT DIR NOT MATCHING")
        
        return subdirs, questions_json
    
    def _build_evaluation_command(self, model_path: Path, questions_json: str) -> List[str]:
        """Build the evaluation command with all parameters."""
        return [
            "python", "expanded_eval.py",
            "--model-name", str(model_path),
            "--generate",
            "--device", "cuda",
            "--ground-truth", self.ground_truth_files,
            "--predictions", "output/predictions",
            "--questions-json", questions_json,
            "--max-new-tokens", "2048",
            "--temperature", "0.7",
            "--do-sample",
            "--bleu",
            "--bleu-type", "all",
            "--rouge",
            "--bert-score",
            "--bert-model", "distilbert-base-uncased",
            "--bert-lang", "de",
            "--deepeval",
        ]
    
    def _get_environment(self) -> Dict[str, str]:
        """Get environment variables for the subprocess."""
        env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        return {**os.environ, **env}
    
    def _cleanup_process(self, process: subprocess.Popen) -> None:
        """Clean up a running process and its children."""
        try:
            # Send SIGTERM to the process group (kills all children too)
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception as e:
            print(f"Could not terminate subprocess group: {e}")
        
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("Force killing subprocess group...")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    
    def _evaluate_single_model(self, model_path: Path, questions_json: str) -> bool:
        """Evaluate a single model and return success status."""
        start_time = time()
        print(f"Evaluating: {model_path}")
        
        command = self._build_evaluation_command(model_path, questions_json)
        
        try:
            process = subprocess.Popen(
                command,
                env=self._get_environment(),
                start_new_session=True,
                stdout=sys.stdout,
                stderr=sys.stderr
            )
            process.wait()
            
            if process.returncode != 0:
                print(f"Evaluation failed for {model_path} with return code {process.returncode}")
                return False
            
            print(f"Finished evaluation for {model_path}")
            elapsed = datetime.timedelta(seconds=time() - start_time)
            print(f"Elapsed time: {elapsed}")
            return True
            
        except KeyboardInterrupt:
            print("\nInterrupt received. Killing subprocess and its children...")
            self._cleanup_process(process)
            sys.exit(130)  # Exit with standard Ctrl+C code
            
        except Exception as e:
            print(f"Unexpected error while evaluating {model_path}: {e}")
            self._cleanup_process(process)
            return False
    
    def evaluate_all_models(self) -> None:
        """Evaluate all models in the configured directories."""
        subdirectories, questions_json = self._get_subdirectories_and_questions()
        
        for subdir in subdirectories:
            success = self._evaluate_single_model(subdir, questions_json)
            if not success:
                sys.exit(1)


def main() -> None:
    """Main entry point for the evaluation script."""
    evaluator = ModelEvaluator(OUTPUT_DIR, GROUND_TRUTH_FILES)
    evaluator.evaluate_all_models()


if __name__ == "__main__":
    main()
