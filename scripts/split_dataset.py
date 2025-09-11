"""
Split a cleaned EGFR CSV into train/valid/test SMILES lists.
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to cleaned CSV")
    ap.add_argument("--smiles_col", default="canonical_smiles", help="SMILES column name")
    ap.add_argument("--label_col", default="pIC50", help="Label column (used for filtering)")
    ap.add_argument("--min_pic50", type=float, default=6.0, help="Min pIC50 to keep for training")
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--valid_size", type=float, default=0.1)
    ap.add_argument("--out_prefix", required=True, help="Output prefix, e.g. data/egfr")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    assert args.smiles_col in df.columns, f"Missing column: {args.smiles_col}"
    assert args.label_col in df.columns, f"Missing column: {args.label_col}"

    # Basic filtering
    df = df.dropna(subset=[args.smiles_col, args.label_col])
    df = df[df[args.label_col].between(3.0, 12.0)]
    potent = df[df[args.label_col] >= args.min_pic50].copy()

    trainval, test = train_test_split(potent, test_size=args.test_size, random_state=42)
    rel_valid = args.valid_size / (1.0 - args.test_size)
    train, valid = train_test_split(trainval, test_size=rel_valid, random_state=42)

    for split_name, split_df in [("train", train), ("valid", valid), ("test", test)]:
        out_path = f"{args.out_prefix}_{split_name}.txt"
        split_df[args.smiles_col].to_csv(out_path, index=False, header=False)
        print(f"Wrote {out_path}: {len(split_df)} SMILES")


if __name__ == "__main__":
    main()

