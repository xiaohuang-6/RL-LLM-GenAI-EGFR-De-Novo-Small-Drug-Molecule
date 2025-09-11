"""
Download EGFR activities from ChEMBL, clean, standardize, compute pIC50, and write CSV + SMILES splits.
Requires network and chembl-webresource-client. If you already have a cleaned CSV,
use `split_dataset.py` directly.
"""
import argparse
import math
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def to_molar(val: float, unit: str) -> float:
    u = (unit or "").lower().strip()
    factor = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "nm": 1e-9, "pm": 1e-12}.get(u, None)
    if factor is None:
        # try simple mapping
        if u in ("molar",):
            factor = 1.0
        else:
            return float("nan")
    return float(val) * factor


def standardize_smiles(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    # Keep largest fragment, neutralize
    lf = rdMolStandardize.LargestFragmentChooser()
    mol = lf.choose(mol)
    un = rdMolStandardize.Uncharger()
    mol = un.uncharge(mol)
    smi_std = Chem.MolToSmiles(mol, canonical=True)
    return smi_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", default="data/all_egfr_ic50_data_cleaned.csv")
    ap.add_argument("--min_pic50", type=float, default=3.0)
    ap.add_argument("--max_pic50", type=float, default=12.0)
    args = ap.parse_args()

    targets = new_client.target
    res = targets.search("EGFR")
    if not res:
        raise RuntimeError("No EGFR targets found via ChEMBL client.")
    tgt_ids = [r["target_chembl_id"] for r in res]

    activities = new_client.activity
    frames = []
    for tid in tgt_ids:
        acts = activities.filter(target_chembl_id=tid, standard_type="IC50")
        frames.append(pd.DataFrame(acts))
    df = pd.concat(frames, ignore_index=True)
    df = df[["molecule_chembl_id", "canonical_smiles", "standard_value", "standard_units"]].dropna()

    # Convert to molar, then pIC50
    df["M"] = df.apply(lambda r: to_molar(r["standard_value"], r["standard_units"]), axis=1)
    df = df.dropna(subset=["M"])  # unknown units removed
    df["pIC50"] = -df["M"].apply(lambda x: math.log10(x) if x > 0 else float("nan"))
    df = df.dropna(subset=["pIC50"])  # remove zeros/invalid
    df = df[df["pIC50"].between(args.min_pic50, args.max_pic50)]

    # Standardize SMILES
    df["canonical_smiles"] = df["canonical_smiles"].apply(standardize_smiles)
    df = df.dropna(subset=["canonical_smiles"]).drop_duplicates(subset=["canonical_smiles", "pIC50"]).reset_index(drop=True)

    out = args.out_csv
    df.to_csv(out, index=False)
    print(f"Wrote cleaned CSV: {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()

