"""
Script to fine-tune an LLM with LoRA using different ranks and alphas. Meant to be started from grid_train.py
"""
import os
# only working if started from grid_train.py else load the .env
from src.util.args import (MODEL_NAME, 
                           PDF_DATA_PATH as DATA_PATH, 
                           PDF_OUTPUT_DIR as OUTPUT_DIR, 
                           MAX_SEQ_LENGTH, TARGET_MODULES,
                           RANK, ALPHA,)
import torch
from unsloth import FastLanguageModel
from transformers.training_args import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

# based on https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb#scrollTo=kR3gIAX-SM2q


def get_pretrained_model_and_tokenizer(load_in_4bit: bool, load_in_8bit: bool, full_finetuning=False):
    """
    Get a pretrained model and tokenizer for fine-tuning from Hugging Face.
    """
    return FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        full_finetuning=full_finetuning
    )


def get_lora_model(model, lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth", use_rslora=False, loftq_config=None):
    """
    Get a LoRA model with specified parameters
    """
    return FastLanguageModel.get_peft_model(
        model,
        r=RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=ALPHA,
        lora_dropout=lora_dropout,  # 0 is optimized
        bias=bias,  # none is optimized
        use_gradient_checkpointing=use_gradient_checkpointing,  # Critical for long context
        use_rslora=use_rslora,
        loftq_config=loftq_config,
        random_state=3407
    )


def get_alpaca_prompt():
    """
    Return the alpaca style prompt for training
    """
    return """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def formatting_prompts_func(examples, alpaca_prompt, eos_token):
    """
    Format dataset to Alpaca style with instruction, input, output
    """
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + eos_token
        texts.append(text)
    return {"text": texts, }


def get_train_args(per_device_train_batch_size=2,
                   gradient_accumulation_steps=4,
                   warmup_steps=10,
                   num_train_epochs=3,
                   # max_steps=1500,
                   learning_rate=1e-4,
                   fp16=not torch.cuda.is_bf16_supported(),
                   bf16=torch.cuda.is_bf16_supported(),
                   logging_steps=50,
                   optim="adamw_8bit",
                   output_dir="outputs",
                   seed=3407,  # fixed seed for reproduceability
                   ):
    """
    Hyperparameters recommendations from: https://docs.unsloth.ai/get-started/fine-tuning-guide#id-6.-training--evaluation
    """
    return TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        # max_steps=1500,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        logging_steps=logging_steps,
        optim=optim,
        output_dir=output_dir,
        seed=seed
    )


def get_sfttrainer(model, tokenizer, formatted_dataset, args, dataset_text_field="text", dataset_num_processes=2, packing=False):
    """
    Get a Supervised Fine-Tuning Trainer with specified args
    """
    return SFTTrainer(model=model,
                      tokenizer=tokenizer,
                      train_dataset=formatted_dataset,
                      dataset_text_field=dataset_text_field,
                      max_seq_length=MAX_SEQ_LENGTH,
                      args=args,
                      dataset_num_proc=dataset_num_processes,  # num of processes to process dataset
                      packing=packing  # Not needed for Alpaca format -> groups sequences into fixed length
                      )


def main():
    """
    Main training function
    """
    model, tokenizer = get_pretrained_model_and_tokenizer(
        load_in_4bit=True, load_in_8bit=False)

    model = get_lora_model(model)

    alpaca_prompt = get_alpaca_prompt()

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    dataset = load_dataset(
        "json",
        data_files=DATA_PATH + "*",  # * needed for dir to not include the glob pattern for eval script
        split="train"
    )

    formatted_dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        fn_kwargs={
            "alpaca_prompt": alpaca_prompt,
            "eos_token": EOS_TOKEN
        }
    )

    train_args = get_train_args()

    trainer = get_sfttrainer(model=model, tokenizer=tokenizer,
                             formatted_dataset=formatted_dataset, args=train_args)

    test_sample = trainer.train_dataset[0]["text"]
    print("Training sample format:\n", test_sample)

    # generation test
    inputs = tokenizer(test_sample, return_tensors="pt").to(model.device)
    print("Output sanity check:\n", tokenizer.decode(
        model.generate(**inputs, max_new_tokens=20)[0]))

    # setup variables for monitoring from https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-CPT.ipynb#scrollTo=2ejIt2xSNKKp&line=1&uniqifier=1
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(
        torch.cuda.max_memory_reserved() / 1024 / 1024 , 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 , 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} MB.")
    print(f"{start_gpu_memory} MB of memory reserved.")

    # get stats from training
    trainer_stats = trainer.train()

    # from https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-CPT.ipynb#scrollTo=pCqnaKmlO1U9&line=3&uniqifier=1
    used_memory = round(torch.cuda.max_memory_reserved() /
                        1024 / 1024 , 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(
        f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} MB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} MB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(
        f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Write to file
    with open(OUTPUT_DIR + "/gpu_usage.txt", "w", encoding="utf-8") as f:
        f.write(
            f"""Peak reserved memory = {used_memory} MB.
Peak reserved memory for training = {used_memory_for_lora} MB.
Peak reserved memory % of max memory = {used_percentage} %.
Peak reserved memory for training % of max memory = {lora_percentage} %.
{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.""")

    # save only adapter to save disk space save_pretrained_merged(save_method="lora") saved whole model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == '__main__':
    main()
