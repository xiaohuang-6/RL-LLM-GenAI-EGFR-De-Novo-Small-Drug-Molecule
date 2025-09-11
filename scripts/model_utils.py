"""
Model and dataset utilities for ChemBERTa-based SMILES modeling.
"""
from typing import List, Optional
import random

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


def load_tokenizer_and_model(model_name_or_path: str = "seyonec/ChemBERTa-zinc-base-v1",
                             device: Optional[str] = None):
    """Load ChemBERTa tokenizer and MaskedLM model.

    Args:
        model_name_or_path: HF model id or local path.
        device: Optional device string (e.g., "cuda" or "cpu"). If None, auto-selects.
    Returns:
        tokenizer, model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model


def mask_random_token(inputs, tokenizer, avoid_special=True):
    """Mask a single random non-special token per sequence in a batch.

    Args:
        inputs: dict with input_ids and attention_mask (torch tensors).
        tokenizer: tokenizer with mask token.
        avoid_special: if True, skip [CLS]/[SEP]/pad tokens.
    Returns:
        masked_inputs, mask_positions
    """
    input_ids = inputs["input_ids"].clone()
    attention_mask = inputs["attention_mask"]
    bsz, seqlen = input_ids.shape
    mask_positions: List[int] = []

    special_ids = set(tokenizer.all_special_ids) if avoid_special else set()
    for i in range(bsz):
        valid_positions = [j for j in range(seqlen)
                           if attention_mask[i, j] == 1 and int(input_ids[i, j]) not in special_ids]
        if not valid_positions:
            mask_positions.append(-1)
            continue
        pos = random.choice(valid_positions)
        input_ids[i, pos] = tokenizer.mask_token_id
        mask_positions.append(pos)
    masked = {k: v.clone() for k, v in inputs.items()}
    masked["input_ids"] = input_ids
    return masked, mask_positions


def sample_replacement_logits(logits, mask_positions):
    """Gather logits at mask positions for sampling.

    Args:
        logits: [B, T, V]
        mask_positions: list of mask indices per batch (or -1)
    Returns:
        gathered [B, V]
    """
    bsz, seqlen, vocab = logits.shape
    device = logits.device
    out = torch.zeros((bsz, vocab), device=device)
    for i, pos in enumerate(mask_positions):
        if pos >= 0:
            out[i] = logits[i, pos]
    return out


def decode_batch(tokenizer, input_ids: torch.Tensor) -> List[str]:
    """Decode batch of token IDs to text (SMILES)."""
    return tokenizer.batch_decode(input_ids, skip_special_tokens=True)

