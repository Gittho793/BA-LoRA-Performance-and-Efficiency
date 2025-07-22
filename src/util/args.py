import os

RANK = int(os.getenv("RANK", "16"))  # as sting for default of getenv
ALPHA = int(os.getenv("ALPHA", "16"))
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
DATA_PATH = "../../data/splitted_pdfs/"
GROUND_TRUTH_FILES = "../../data/raw_splitted_pdfs"
OUTPUT_DIR = f"../../../unslothLora/Llama/pdf-ds-new-llama-3.1-8b-v0.3-4bit-lora-r{RANK}-a{ALPHA}"
TXT_DATA_PATH = "../../data/splitted_txts/"
TXT_GROUND_TRUTH_FILES = "../../data/raw_splitted_txt"
TXT_OUTPUT_DIR = f"../../../unslothLora/Llama/txt-ds-new-llama-3.1-8b-v0.3-4bit-lora-r{RANK}-a{ALPHA}"
MAX_SEQ_LENGTH: int = 4096
TARGET_MODULES = ["q_proj", "k_proj", "v_proj",
                  "o_proj", "gate_proj", "up_proj", "down_proj"]
