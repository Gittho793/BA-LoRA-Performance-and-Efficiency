"""
Evaluate multiple models from one directory.
"""
import signal
import subprocess
import os
import sys
import datetime
from time import time
from dotenv import load_dotenv
from src.util.args import (PDF_OUTPUT_DIR as OUTPUT_DIR,
                           PDF_GROUND_TRUTH_FILES as GROUND_TRUTH_FILES)


load_dotenv("../../.env")

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


BASE_DIR = os.path.dirname(os.path.abspath(OUTPUT_DIR))

if "txt" in OUTPUT_DIR and "txt" in GROUND_TRUTH_FILES:
    top_level_subdirs = [
        os.path.join(BASE_DIR, name)
        for name in sorted(os.listdir(BASE_DIR))
        if name.startswith("txt") and os.path.isdir(os.path.join(BASE_DIR, name))
    ]
    questions_json = "output/txt_questions.json"
elif "pdf" in OUTPUT_DIR and "pdf" in GROUND_TRUTH_FILES:
    top_level_subdirs = [
        os.path.join(BASE_DIR, name)
        for name in sorted(os.listdir(BASE_DIR))
        if name.startswith("pdf") and os.path.isdir(os.path.join(BASE_DIR, name))
    ]
    questions_json = "output/pdf_questions.json"
else:
    raise ValueError("GROUND TRUTH AND OUTPUT DIR NOT MATCHING")

env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

for subdir in top_level_subdirs:
    start = time()
    print("Evaluating:", subdir)

    command = [
        "python", "expanded_eval.py",
        "--model-name", subdir,
        "--generate",
        "--device", "cuda",
        "--ground-truth", GROUND_TRUTH_FILES,
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

    try:
        process = subprocess.Popen(
            command,
            env={**os.environ, **env},
            start_new_session=True,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        process.wait()

        if process.returncode != 0:
            print(f"Evaluation failed for {subdir} with return code {process.returncode}")
            sys.exit(process.returncode)

        print(f"Finished evaluation for {subdir}")
        elapsed = time() - start
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed time: ", elapsed)

    except KeyboardInterrupt:
        print("\nInterrupt received. Killing subprocess and its children...")

        try:
            # Send SIGTERM to the *process group* (kills all children too)
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception as e:
            print(f"Could not terminate subprocess group: {e}")

        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f"Force killing subprocess group...")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)

        sys.exit(130)  # Exit with standard Ctrl+C code

    except Exception as e:
        print(f"Unexpected error while evaluating {subdir}: {e}")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        sys.exit(1)
