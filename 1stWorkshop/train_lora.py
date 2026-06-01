import torch
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, DatasetDict


def build_prompt(example):
    instruction = example["instruction"].strip()
    user_input = example["input"].strip()
    output = example["output"].strip()

    if user_input:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{user_input}\n\n"
            f"### Response:\n{output}"
        )

    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def tokenize_function(example, tokenizer):
    prompt = build_prompt(example)
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


class MultiEvalTrainer(Trainer):
    """Trainer that also evaluates the test set at every eval step."""

    def __init__(self, test_dataset=None, **kwargs):
        super().__init__(**kwargs)
        self._test_dataset = test_dataset

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        # Only run test eval during training-loop evals, not on explicit external calls
        if eval_dataset is None and self._test_dataset is not None:
            test_metrics = super().evaluate(
                eval_dataset=self._test_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix="test",
            )
            self.log({"test_loss": test_metrics["test_loss"]})
        return metrics


def main():
    # Change this to switch the base model easily.
    # Example options:
    #   "Qwen/Qwen3.5-27B"
    #   "Qwen/Qwen3-32B"
    #   "google/gemma-3-27b-it"
    #   "mistralai/Mistral-7B-Instruct-v0.3"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print("Loading dataset...")
    # --- Option A (current): load a public HuggingFace dataset ---
    raw_dataset = load_dataset("yahma/alpaca-cleaned")
    small_dataset = raw_dataset["train"].shuffle(seed=42).select(range(600))

    # --- Option B: load a local JSON/CSV file and split it yourself ---
    # from datasets import load_dataset
    # raw_dataset = load_dataset("json", data_files="path/to/your_data.json")
    # # or for CSV:
    # # raw_dataset = load_dataset("csv", data_files="path/to/your_data.csv")
    #
    # # Split into train / validation / test manually:
    # train_test  = raw_dataset["train"].train_test_split(test_size=0.20, seed=42)
    # valid_test  = train_test["test"].train_test_split(test_size=0.50, seed=42)
    # data = DatasetDict({
    #     "train":      train_test["train"],
    #     "validation": valid_test["train"],
    #     "test":       valid_test["test"],
    # })

    train_test = small_dataset.train_test_split(test_size=0.20, seed=42)
    valid_test = train_test["test"].train_test_split(test_size=0.50, seed=42)

    data = DatasetDict({
        "train": train_test["train"],
        "validation": valid_test["train"],
        "test": valid_test["test"],
    })

    print("Preparing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = data.map(
        lambda example: tokenize_function(example, tokenizer),
        batched=False,
    )

    print("Loading base model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    base_model = prepare_model_for_kbit_training(base_model)

    # The task_type tells PEFT what kind of problem we are solving.
    # Common values include:
    # - CAUSAL_LM: decoder-only text generation models like GPT, LLaMA, Mistral.
    # - SEQ_2_SEQ_LM: encoder-decoder models like T5 or BART.
    # - SEQ_CLASSIFICATION: sentence/document classification.
    # - TOKEN_CLS: token-level classification tasks such as named entity recognition.
    # - QUESTION_ANSWERING: QA tasks with answer generation or span prediction.
    # We use CAUSAL_LM for this decoder-only instruction-following model.
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="workshop_outputs/lora-alpaca",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=5,
        save_steps=5,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        optim="paged_adamw_32bit",
        report_to="none",
    )

    trainer = MultiEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        test_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # --- collect losses from log history ---
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    test_steps, test_losses = [], []

    for entry in trainer.state.log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])
        if "test_loss" in entry:
            test_steps.append(entry["step"])
            test_losses.append(entry["test_loss"])

    # --- plot ---
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(train_steps, train_losses, label="Train loss", color="steelblue", linewidth=2)
    ax.plot(eval_steps, eval_losses, label="Validation loss", color="darkorange",
            linewidth=2, marker="o", markersize=5)
    ax.plot(test_steps, test_losses, label="Test loss", color="crimson",
            linewidth=2, marker="s", markersize=5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("LoRA Fine-Tuning — Train / Validation / Test Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = "workshop_outputs/lora-alpaca/loss_curves.png"
    fig.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Loss plot saved to {plot_path}")

    trainer.save_model("lora_alpaca_adapter")
    tokenizer.save_pretrained("lora_alpaca_adapter")
    print("Saved LoRA adapter to lora_alpaca_adapter")


if __name__ == "__main__":
    main()
