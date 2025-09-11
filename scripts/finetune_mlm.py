"""
Fine-tune ChemBERTa via masked language modeling (MLM) on EGFR SMILES.
"""
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to train SMILES txt")
    ap.add_argument("--valid", required=True, help="Path to valid SMILES txt")
    ap.add_argument("--model_name_or_path", default="seyonec/ChemBERTa-zinc-base-v1")
    ap.add_argument("--out_dir", default="models/chemberta_mlm_ft")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_length", type=int, default=128)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)

    # Load text files as datasets
    ds_train = load_dataset("text", data_files={"train": args.train})["train"]
    ds_valid = load_dataset("text", data_files={"validation": args.valid})["validation"]

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=args.max_length)

    ds_train = ds_train.map(tok_fn, batched=True, remove_columns=["text"])
    ds_valid = ds_valid.map(tok_fn, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
    )
    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)


if __name__ == "__main__":
    main()

