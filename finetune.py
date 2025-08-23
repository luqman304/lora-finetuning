import argparse, os, json, torch
from pathlib import Path

def load_toy(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main(args):
    if args.eval_only:
        print("Eval stub. Load from", args.out_dir, "and compute exact‑match/ROUGE for your task."); return

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except Exception:
        print("Install dependencies: pip install transformers datasets accelerate peft"); raise

    use_lora = False
    try:
        from peft import get_peft_model, LoraConfig, TaskType
        use_lora = True
    except Exception:
        print("PEFT not found; running without LoRA. (pip install peft)")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    if use_lora:
        peft_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05)
        model = get_peft_model(model, peft_cfg)

    data = load_toy("data/toy.jsonl")
    def fmt(ex): return f"### Instruction:\\n{ex['instruction']}\\n### Input:\\n{ex['input']}\\n### Output:"
    inputs = [fmt(d) for d in data]; labels = [d["output"] for d in data]

    enc = tok(inputs, padding=True, truncation=True, return_tensors="pt")
    with tok.as_target_tokenizer():
        dec = tok(labels, padding=True, truncation=True, return_tensors="pt")
    enc["labels"] = dec["input_ids"]

    class DS(torch.utils.data.Dataset):
        def __init__(self, enc): self.enc = enc
        def __len__(self): return self.enc["input_ids"].shape[0]
        def __getitem__(self, i): 
            return {k: v[i] for k,v in self.enc.items()}

    train_ds = DS(enc)
    from transformers import default_data_collator
    args_out = TrainingArguments(output_dir=args.out_dir, per_device_train_batch_size=args.batch_size,
                                 num_train_epochs=args.epochs, learning_rate=5e-4, logging_steps=5,
                                 save_steps=50, fp16=torch.cuda.is_available(), report_to=[])
    trainer = Trainer(model=model, args=args_out, data_collator=default_data_collator, train_dataset=train_ds)
    trainer.train()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.out_dir); tok.save_pretrained(args.out_dir)
    print("Saved to", args.out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=str, default="sshleifer/tiny-gpt2")
    ap.add_argument("--out-dir", type=str, default="outputs/tiny-gpt2-lora")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args(); main(args)
