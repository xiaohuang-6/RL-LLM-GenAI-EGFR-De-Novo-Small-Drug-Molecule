"""
Prepare 3D structures from SMILES and optionally export PDBQT for docking.
Requires RDKit (for embedding) and optionally OpenBabel (`obabel`) for PDBQT.
"""
import argparse
import os
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem


def write_sdf(mol, path):
    w = Chem.SDWriter(path)
    w.write(mol)
    w.flush()
    w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="SMILES txt file")
    ap.add_argument("--out_dir", required=True, help="Output directory for 3D ligands")
    ap.add_argument("--to_pdbqt", action="store_true", help="Export PDBQT via OpenBabel if available")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.input) as f:
        smiles = [l.strip() for l in f if l.strip()]

    for i, smi in enumerate(smiles, start=1):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            continue
        base = os.path.join(args.out_dir, f"lig{i:04d}")
        sdf_path = base + ".sdf"
        write_sdf(mol, sdf_path)
        if args.to_pdbqt:
            pdbqt_path = base + ".pdbqt"
            try:
                subprocess.run(["obabel", sdf_path, "-O", pdbqt_path, "--partialcharge", "gasteiger"], check=True)
            except Exception as e:
                print(f"OpenBabel conversion failed for {sdf_path}: {e}")

    print(f"Prepared 3D ligands in {args.out_dir}")


if __name__ == "__main__":
    main()

