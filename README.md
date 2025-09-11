# EGFR Generative Pipeline (ChemBERTa + RL)

This repository organizes a complete, scriptable pipeline for EGFR-focused de novo small-molecule generation, RL fine-tuning with a composite reward (potency, QED, SA, novelty), and docking-based validation.

The code is modular: each stage can be run independently. Where network or heavy dependencies are required (e.g., Hugging Face model downloads, RDKit, docking binaries), the scripts assume a properly configured environment. Placeholders are provided for data, models, and figures.

## Directory Layout

- `scripts/`
  - `data_download_and_clean.py`: Download EGFR data from ChEMBL, clean, and export `all_egfr_ic50_data_cleaned.csv` and SMILES splits.
  - `split_dataset.py`: Split a cleaned CSV into train/valid/test SMILES lists.
  - `model_utils.py`: Load/save ChemBERTa tokenizer and model utilities.
  - `finetune_mlm.py`: Masked language modeling fine-tune for ChemBERTa on EGFR SMILES.
  - `train_qsar_rf.py`: Train RF (Morgan fingerprints → pIC50) and save `egfr_rf_qsar.pkl`.
  - `rl_finetune.py`: RL fine-tune with composite reward (potency, QED, SA, novelty).
  - `generate_molecules.py`: Sample molecules from (RL-)tuned model, deduplicate, validate, and save.
  - `prepare_3d_ligands.py`: Optional helper to embed SMILES to 3D (SDF) and export to PDBQT (requires OpenBabel/MGLTools).
  - `analyze_results.py`: Aggregate stats (validity/uniqueness/novelty/diversity) and plot distributions.
- `docking/`
  - `docking.sh`: Loops for AutoDock Vina docking on original and generated ligands.
  - `EXTRACT.py`: Parse vina text logs into a CSV (`docking_scores.csv`).
- `data/`
  - `all_egfr_ic50_data_cleaned.csv` (symlink or copy from project root; see below)
  - `egfr_train.txt`, `egfr_valid.txt`, `egfr_test.txt` (SMILES lists).
- `models/`
  - `chemberta_prior/` (downloaded or cached pretrained prior).
  - `chemberta_mlm_ft/` (MLM fine-tuned checkpoint).
  - `chemberta_rl_ft/` (RL fine-tuned checkpoint).
  - `egfr_rf_qsar.pkl` (RF oracle; placeholder until trained).
- `ligands/`
  - `original_ligand_3d/` (pdbqt files for reference/top EGFR ligands; keep outputs here).
  - `generated_ligand_3d/` (pdbqt files for de novo molecules; keep outputs here).
- `results/`
  - Generated SMILES, plots, docking tables.
- `figures/`
  - Project diagrams/plots (placeholders included).

## Environment

- Python 3.9+
- Packages: see `requirements.txt`
- External tools:
  - AutoDock Vina 1.2.7+ (binary on PATH or in project root)
  - OpenBabel (optional, for PDBQT prep) and/or MGLTools scripts (`prepare_ligand4.py`)
  - RDKit (must be installed per OS instructions)

Install requirements (CPU example):

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

1) Data (if you already have a cleaned CSV):
- Place `all_egfr_ic50_data_cleaned.csv` at project root or copy into `egfr_pipeline/data/`.
- Create SMILES splits:
```
python scripts/split_dataset.py \
  --input data/all_egfr_ic50_data_cleaned.csv \
  --smiles_col canonical_smiles --label_col pIC50 \
  --out_prefix data/egfr
```
This writes `egfr_train.txt`, `egfr_valid.txt`, `egfr_test.txt` into `data/`.

2) Download/load prior model:
- By default uses `seyonec/ChemBERTa-zinc-base-v1`. The scripts can run offline if the model/tokenizer are cached locally (set `--model_name_or_path models/chemberta_prior`).

3) MLM fine-tune ChemBERTa:
```
python scripts/finetune_mlm.py \
  --train data/egfr_train.txt --valid data/egfr_valid.txt \
  --out_dir models/chemberta_mlm_ft
```

4) QSAR oracle (Random Forest):
```
python scripts/train_qsar_rf.py \
  --input data/all_egfr_ic50_data_cleaned.csv \
  --smiles_col canonical_smiles --label_col pIC50 \
  --out models/egfr_rf_qsar.pkl
```

5) RL fine-tune with composite reward:
```
python scripts/rl_finetune.py \
  --train data/egfr_train.txt --valid data/egfr_valid.txt \
  --rf_model models/egfr_rf_qsar.pkl \
  --out_dir models/chemberta_rl_ft
```

6) Generate molecules (100 new):
```
python scripts/generate_molecules.py \
  --model models/chemberta_rl_ft --num 100 \
  --out_smiles results/generated_100.txt
```

7) Prepare 3D and PDBQT (optional helper):
```
python scripts/prepare_3d_ligands.py \
  --input results/generated_100.txt \
  --out_dir ligands/generated_ligand_3d
```
Repeat for your original/reference ligands into `ligands/original_ligand_3d`.

8) Docking with Vina:
- Place receptor `1M17_clean.pdbqt` into project root or `docking/`.
- Adjust paths in `docking/docking.sh` if needed; then run:
```
bash docking/docking.sh
```

9) Extract docking scores:
```
python docking/EXTRACT.py \
  --dirs ligands/original_ligand_3d ligands/generated_ligand_3d \
  --out results/docking_scores.csv
```

10) Analyze results:
```
python scripts/analyze_results.py \
  --docking_csv results/docking_scores.csv \
  --out_dir results
```

## Notes

- Placeholders provided: `models/egfr_rf_qsar.pkl` (empty until trained), `figures/` and `.gitkeep` files, empty ligand folders. Replace with real outputs after running the pipeline.
- If you are fully offline, download the Hugging Face model in advance or provide a local path via `--model_name_or_path`.
- Docking requires prepared PDBQT ligands and receptor; ensure correct center and box parameters for Vina.

## References
- ChemBERTa: Ahmad et al., 2022.
- RDKit: Landrum, 2025.
- AutoDock Vina 1.2.7: Eberhardt et al., 2021.
- EGFR PDB 1M17: Stamos et al., 2002.

