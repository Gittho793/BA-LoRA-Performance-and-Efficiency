'''
Train a LLM with LoRA using different ranks and alphas. Alpha is either the same or double the rank.
'''
import subprocess
import os
import sys
from dotenv import load_dotenv

load_dotenv("../../.env")  # necessary for local imports on cluster

ranks: list[int] = [4, 8, 16, 32, 64, 128]
if __name__ == "__main__":
    for rank in ranks:

        for i in range(1, 3, 1):
            alpha = rank * i
            print(f"=Training with RANK={rank}, ALPHA={alpha}")
            env = {
                "RANK": str(rank),
                "ALPHA": str(alpha),
            }
            try:
                # train.py with overridden environment variables
                result = subprocess.run(["python", "train.py"], env={
                                        **env, **dict(os.environ)},
                                        check=True,
                                        start_new_session=True,
                                        stdout=sys.stdout,
                                        stderr=sys.stderr)

                print(f"Finished RANK={rank}, ALPHA={alpha}")

            except subprocess.CalledProcessError as e:
                print(f"Training failed for RANK={rank}, ALPHA={alpha}")
                sys.exit()
                