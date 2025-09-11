"""
Train a LLM with LoRA using different ranks and alphas. 
Alpha is either the same or double the rank.
"""
import subprocess
import os
import sys
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv("../../.env")  # necessary for local imports on cluster in train.py

RANKS: List[int] = [4, 8, 16, 32, 64, 128]
ALPHA_MULTIPLIERS: List[int] = [1, 2]


def run_training(rank: int, alpha: int) -> bool:
    """
    Run training with specified rank and alpha values.

    Args:
        rank: LoRA rank parameter
        alpha: LoRA alpha parameter

    Returns:
        True if training succeeded, False otherwise
    """
    print(f"=Training with RANK={rank}, ALPHA={alpha}")

    env: Dict[str, str] = {
        "RANK": str(rank),
        "ALPHA": str(alpha),
    }

    try:
        _ = subprocess.run(
            ["python", "train.py"],
            env={**env, **dict(os.environ)},
            check=True,
            start_new_session=True,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        print(f"Finished RANK={rank}, ALPHA={alpha}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Training failed for RANK={rank}, ALPHA={alpha}: {e}")
        return False


def main() -> None:
    """Main training loop for grid search over rank and alpha parameters."""
    for rank in RANKS:
        for multiplier in ALPHA_MULTIPLIERS:
            alpha = rank * multiplier

            success = run_training(rank, alpha)
            if not success:
                print("Stopping grid search due to training failure")
                sys.exit(1)


if __name__ == "__main__":
    main()
