from src.util.args import TXT_OUTPUT_DIR, TXT_GROUND_TRUTH_FILES
import subprocess
import signal
import os
import sys
from dotenv import load_dotenv

load_dotenv("../../.env")

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


BASE_DIR = os.path.dirname(os.path.abspath(TXT_OUTPUT_DIR))


top_level_subdirs = [
    os.path.join(BASE_DIR, name)
    for name in sorted(os.listdir(BASE_DIR))
    if name.startswith("txt") and "8bit" in name and os.path.isdir(os.path.join(BASE_DIR, name))
]
env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

extract_questions = True

for subdir in top_level_subdirs:
    print("Evaluating:", subdir)

    command = [
        "python", "expanded_eval.py",
        "--model-name", subdir,
        # "--generate",
        "--device", "cuda",
        "--ground-truth", TXT_GROUND_TRUTH_FILES,
        "--predictions", "output/predictions",
        "--max-new-tokens", "1024",
        "--temperature", "0.7",
        "--do-sample",
        # "--bleu",
        # "--bleu-type", "all",
        # "--rouge",
        # "--bert-score",
        # "--bert-model", "distilbert-base-uncased",
        # "--bert-lang", "de",
        "--extracted-questions", "output/questions",
        "--deepeval",
        "--use_questions"
    ]

    if extract_questions:
        # to ensure the llms are evaluated on the same questions
        # command.append("--extract-questions")
        extract_questions = False

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
        break

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
        print(f"🔥 Unexpected error while evaluating {subdir}: {e}")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        sys.exit(1)
