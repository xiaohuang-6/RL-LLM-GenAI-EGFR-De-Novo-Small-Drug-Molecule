"""
Reinforcement Learning fine-tuning for ChemBERTa via a simple policy-gradient
approach that masks one token in a SMILES and samples a replacement. Reward is a
weighted sum of predicted potency (RF model on Morgan FP), QED, SA (inverted),
and novelty vs training set.
"""
import argparse
import random
import math
from collections import deque
from typing import List

import joblib
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from transformers import AutoTokenizer, AutoModelForMaskedLM

from model_utils import mask_random_token, sample_replacement_logits, decode_batch


def morgan_fp(mol, n_bits: int = 2048, radius: int = 2):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    # noinspection PyUnresolvedReferences
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def sa_score(mol):
    # Simple proxy using RDKit SA_Score implementation if available; fallback to ring complexity
    try:
        from rdkit.Chem import rdMolDescriptors as desc
        sa = desc.CalcNumAtomStereoCenters(mol) + desc.CalcNumBridgeheadAtoms(mol) + desc.CalcNumSpiroAtoms(mol)
        # this is not the canonical Ertl SA; user may plug in their own function
        return float(sa) if sa > 0 else 3.0
    except Exception:
        return 3.0


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def normalize_pic50(pic50: float) -> float:
    return max(0.0, min(1.0, (pic50 - 5.0) / 5.0))


def invert_sa(sa: float) -> float:
    return max(0.0, min(1.0, (10.0 - sa) / 9.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Train SMILES txt")
    ap.add_argument("--valid", required=True, help="Valid SMILES txt")
    ap.add_argument("--rf_model", required=True, help="Path to trained RF joblib")
    ap.add_argument("--model_name_or_path", default="seyonec/ChemBERTa-zinc-base-v1")
    ap.add_argument("--out_dir", default="models/chemberta_rl_ft")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    rf = joblib.load(args.rf_model)

    # Load training set SMILES and compute training fingerprints for novelty
    with open(args.train) as f:
        train_smiles = [line.strip() for line in f if line.strip()]
    train_fps = []
    for smi in train_smiles[:5000]:  # cap for memory; adjust as needed
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        train_fps.append(morgan_fp(mol))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    recent_rewards = deque(maxlen=100)

    def tokenize_smiles(batch_smiles: List[str]):
        return tokenizer(batch_smiles, truncation=True, padding=True, max_length=args.max_length, return_tensors="pt").to(device)

    for step in range(1, args.steps + 1):
        batch = random.sample(train_smiles, k=min(args.batch_size, len(train_smiles)))
        inputs = tokenize_smiles(batch)
        masked_inputs, mask_pos = mask_random_token(inputs, tokenizer)

        outputs = model(**masked_inputs)
        logits = outputs.logits  # [B, T, V]
        gathered = sample_replacement_logits(logits, mask_pos)
        probs = torch.softmax(gathered, dim=-1)
        # Sample tokens at the mask position
        sampled_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Replace mask with sampled token
        input_ids = masked_inputs["input_ids"].clone()
        for i, pos in enumerate(mask_pos):
            if pos >= 0:
                input_ids[i, pos] = sampled_ids[i]

        # Compute log-prob of sampled token for REINFORCE loss
        logp = torch.log(torch.gather(probs, 1, sampled_ids.unsqueeze(-1)).squeeze(-1) + 1e-9)  # [B]

        # Decode to SMILES strings
        decoded = decode_batch(tokenizer, input_ids)

        # Compute rewards per sample
        rewards = []
        for smi in decoded:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rewards.append(-1.0)
                continue
            # Potency via RF
            fp = morgan_fp(mol)
            pic50_pred = float(rf.predict(fp.reshape(1, -1))[0])
            r_pic50 = normalize_pic50(pic50_pred)
            # QED and SA
            r_qed = float(QED.qed(mol))
            r_sa = invert_sa(sa_score(mol))
            # Novelty vs training set (1 - max tanimoto)
            max_sim = 0.0
            for tr in train_fps:
                sim = tanimoto(fp, tr)
                if sim > max_sim:
                    max_sim = sim
            r_novel = 1.0 - max_sim
            reward = 0.40 * r_pic50 + 0.25 * r_qed + 0.15 * r_sa + 0.20 * r_novel
            rewards.append(reward)

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        recent_rewards.extend(rewards)

        # Policy gradient objective: maximize E[R * log p]
        loss = -(rewards_t * logp).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 20 == 0:
            avg_r = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
            print(f"Step {step:4d} | loss {loss.item():.4f} | mean R {avg_r:.4f} | max R {max(recent_rewards) if recent_rewards else 0:.4f}")

    # Save checkpoint
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved RL-tuned model to {args.out_dir}")


if __name__ == "__main__":
    main()

