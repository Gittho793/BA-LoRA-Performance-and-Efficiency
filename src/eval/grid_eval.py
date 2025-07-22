from src.util.args import TXT_OUTPUT_DIR, TXT_GROUND_TRUTH_FILES
import subprocess
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
        "--generate",
        "--device", "cuda",
        "--ground-truth", TXT_GROUND_TRUTH_FILES,
        "--predictions", "output/predictions",
        "--max-new-tokens", "1024",
        "--temperature", "0.7",
        "--do-sample",
        "--bleu",
        "--bleu-type", "all",
        "--rouge",
        "--bert-score",
        "--bert-model", "distilbert-base-uncased",
        "--bert-lang", "de",
        "--extracted-questions", "output/questions",
        "--deepeval",
        "--use_questions"
    ]

    if extract_questions:
        # to ensure the llms are evaluated on the same questions
        # command.append("--extract-questions")
        extract_questions = False

    try:
        result = subprocess.run(
            command,
            check=True,
            start_new_session=True,
            env={**env, **dict(os.environ)},
            stdout=sys.stdout,
            stderr=sys.stderr
        )

        print(f"Finished evaluation for {subdir}")
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed for {subdir} with return code {e.returncode}")
        print(f"Command that failed: {e.cmd}")
        sys.exit()
