"""
Simply test the inference of a local LLM using unsloth FastLanguageModel.
"""
import unsloth  # wants to be on top for optimization
from dotenv import load_dotenv
from transformers.generation.streamers import TextStreamer
from unsloth import FastLanguageModel
import os
import sys


load_dotenv("../../../../.env")

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.util.args import MAX_SEQ_LENGTH, PDF_OUTPUT_DIR

# based on https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb#scrollTo=kR3gIAX-SM2q


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="../../"+PDF_OUTPUT_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    load_in_8bit=False,
)

FastLanguageModel.for_inference(model)

text_streamer = TextStreamer(tokenizer, skip_prompt=True)


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# from Schlueter_Kress_2017_Methoden_und_Techniken_chapter_08_qa_pairs_cleaned.json
prompt = "Was ist die Bodenanker-Übung und was beinhaltet sie?"
# prompt = "Was ist Microcounseling?"

# sanity check
# prompt = "The colors of the rainbow are:"


inputs = tokenizer([alpaca_prompt.format(
    "Beantworte die folgende Frage",  # instruction
    prompt,  # input
    ""  # output -> blank for generation
)], return_tensors="pt").to("cuda")


_ = model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=2048,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.2,
    num_beams=1
)
