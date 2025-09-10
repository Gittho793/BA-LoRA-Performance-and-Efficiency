"""
Test tokenization with sliding window approach and print decoded outputs.
Just to ensure tokenization works as expected.
"""
import os
import sys
from datasets import load_dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer  # Use standard tokenizer
from dotenv import load_dotenv

load_dotenv("../../../../.env")

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.util.args import MODEL_NAME, PDF_DATA_PATH, MAX_SEQ_LENGTH

def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)

def sliding_tokenize(examples, tokenizer, stride=256):
    text_column = "text"

    tokenized = tokenizer(
        examples[text_column],
        max_length=MAX_SEQ_LENGTH,
        stride=stride,
        return_overflowing_tokens=True,
        truncation=True,
        padding="max_length"
    )

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "overflow_to_sample_mapping": tokenized["overflow_to_sample_mapping"]
    }

def main():
    tokenizer = load_tokenizer()

    dataset = load_dataset("json", data_files=PDF_DATA_PATH, split="train")

    tokenized_dataset = dataset.map(
        sliding_tokenize,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        remove_columns=dataset.column_names
    )


    # Just print a few examples
    for i in range(3):
        input_ids = tokenized_dataset[i]["input_ids"]
        print(input_ids)
        print(tokenizer.decode(input_ids, skip_special_tokens=True))

    output_path = "tokenized_output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(3):
            input_ids = tokenized_dataset[i]["input_ids"]
            decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            f.write(f"Example {i+1}:\n")
            f.write(decoded_text + "\n\n")

    print(f"\nTokenized and decoded examples saved to: {output_path}")

if __name__ == "__main__":
    main()
