"""
Train a Random Forest QSAR model (Morgan fingerprints → pIC50) and save as joblib.
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def smiles_to_morgan_fp(smiles: str, n_bits: int = 2048, radius: int = 2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    # noinspection PyUnresolvedReferences
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Cleaned CSV with SMILES + pIC50")
    ap.add_argument("--smiles_col", default="canonical_smiles")
    ap.add_argument("--label_col", default="pIC50")
    ap.add_argument("--out", default="models/egfr_rf_qsar.pkl")
    ap.add_argument("--test_size", type=float, default=0.1)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df = df.dropna(subset=[args.smiles_col, args.label_col])
    feats, labels = [], []
    for smi, y in zip(df[args.smiles_col].tolist(), df[args.label_col].tolist()):
        arr = smiles_to_morgan_fp(smi)
        if arr is None:
            continue
        feats.append(arr)
        labels.append(float(y))
    X = np.asarray(feats)
    y = np.asarray(labels)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    ypred = rf.predict(Xte)
    print(f"Test R2: {r2_score(yte, ypred):.3f}")

    joblib.dump(rf, args.out)
    print(f"Saved RF model to {args.out}")


if __name__ == "__main__":
    main()

