"""
Aggregate simple metrics and plots from docking outputs and generated SMILES.
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docking_csv", required=True, help="CSV from EXTRACT.py")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.docking_csv)

    # Expect columns: file, set, score
    print(df.groupby("set")["score"].describe())

    plt.figure(figsize=(7,5))
    sns.kdeplot(data=df, x="score", hue="set", fill=True)
    plt.title("Docking Score Distributions (lower is better)")
    plt.xlabel("Vina score (kcal/mol)")
    plt.tight_layout()
    out_plot = os.path.join(args.out_dir, "docking_score_distribution.png")
    plt.savefig(out_plot, dpi=200)
    print(f"Saved {out_plot}")


if __name__ == "__main__":
    main()

