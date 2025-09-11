"""
Sample molecules from a (RL-)tuned ChemBERTa MLM by iterative single-token edits.
"""
import argparse
import random
from typing import List, Set

import torch
from rdkit import Chem
from transformers import AutoTokenizer, AutoModelForMaskedLM

from model_utils import mask_random_token, sample_replacement_logits, decode_batch


def sample_iterative(tokenizer, model, seed_smiles: List[str], steps: int = 10, max_length: int = 128) -> List[str]:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(seed_smiles, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(device)
        for _ in range(steps):
            masked_inputs, mask_pos = mask_random_token(inputs, tokenizer)
            logits = model(**masked_inputs).logits
            gathered = sample_replacement_logits(logits, mask_pos)
            probs = torch.softmax(gathered, dim=-1)
            sampled_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
            input_ids = masked_inputs["input_ids"].clone()
            for i, pos in enumerate(mask_pos):
                if pos >= 0:
                    input_ids[i, pos] = sampled_ids[i]
            inputs["input_ids"] = input_ids
        return decode_batch(tokenizer, inputs["input_ids"])  # final decoded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to tuned model or HF id")
    ap.add_argument("--num", type=int, default=100, help="Number of molecules to generate")
    ap.add_argument("--seed_smiles", default=None, help="Optional seed SMILES txt; else random from tokenizer vocab")
    ap.add_argument("--out_smiles", default="results/generated_100.txt")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--edit_steps", type=int, default=12)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Seed sequences: start from valid SMILES seeds or simple atoms
    if args.seed_smiles:
        with open(args.seed_smiles) as f:
            seeds = [line.strip() for line in f if line.strip()]
    else:
        seeds = ["CC", "CCC", "c1ccccc1", "CCO", "CCN"]

    out: List[str] = []
    seen: Set[str] = set()
    while len(out) < args.num:
        batch = random.sample(seeds, k=min(32, args.num - len(out), len(seeds)))
        decoded = sample_iterative(tokenizer, model, batch, steps=args.edit_steps, max_length=args.max_length)
        for smi in decoded:
            if smi in seen:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            out.append(smi)
            seen.add(smi)
            if len(out) >= args.num:
                break

    # Write
    import os
    os.makedirs(os.path.dirname(args.out_smiles), exist_ok=True)
    with open(args.out_smiles, "w") as f:
        for smi in out:
            f.write(smi + "\n")
    print(f"Wrote {len(out)} molecules to {args.out_smiles}")


if __name__ == "__main__":
    main()

