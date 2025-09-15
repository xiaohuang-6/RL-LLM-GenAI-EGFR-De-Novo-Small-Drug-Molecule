# EGFR Generative Pipeline (ChemBERTa + RL)

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg) ![ChemBERTa](https://img.shields.io/badge/model-ChemBERTa-success.svg) ![AutoDock Vina](https://img.shields.io/badge/docking-AutoDock%20Vina-orange.svg) ![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

**Design potent small molecules against EGFR end-to-end — from data curation, ChemBERTa pretraining, RL policy optimisation, to AutoDock Vina validation.**

> This project packages the workflow our team uses to bootstrap EGFR-focused molecule generation experiments. Everything is scriptable, reproducible, and ready for scale — no manual notebooks required.

---

## Table of Contents
- [Highlights](#highlights)
- [Why EGFR?](#why-egfr)
- [Pipeline Overview](#pipeline-overview)
- [Project Layout](#project-layout)
- [Getting Started](#getting-started)
- [End-to-End Runbook](#end-to-end-runbook)
- [Result Highlights](#result-highlights)
- [Troubleshooting & Tips](#troubleshooting--tips)
- [Roadmap](#roadmap)
- [Citations](#citations)

## Highlights
- **Modular scripts** for every stage: data prep, ChemBERTa MLM fine-tuning, RL policy optimisation, molecule sampling, analytics, and docking.
- **Composite reward RL** balances potency (RF QSAR oracle), drug-likeness (QED), synthetic accessibility, and novelty.
- **Reproducible docking loop** with AutoDock Vina 1.2.7 and log extraction utilities to benchmark generated ligands.
- **Figure-ready outputs** (plots, tables, PDFs) for reporting progress to collaborators or investors.
- **Offline-first design** — cache Hugging Face checkpoints, run without internet, and keep sensitive data on-prem.

## Why EGFR?
Epidermal Growth Factor Receptor (EGFR) is a clinically validated oncology target with rich public assay data and structural information. This repository focuses on EGFR to:
- Demonstrate how large chemical language models can be specialised via transfer learning.
- Provide a reference RL environment for medicinal chemistry reward shaping.
- Offer docking-ready assets (receptor, ligands) for rapid in-silico validation.

## Pipeline Overview
| Stage | Goal | Script | Key Outputs |
| --- | --- | --- | --- |
| 1. Data Curation | Download + clean EGFR bioactivity records | `scripts/data_download_and_clean.py` | `all_egfr_ic50_data_cleaned.csv` |
| 2. Train/Valid/Test Splits | Balanced SMILES splits | `scripts/split_dataset.py` | `data/egfr_{train,valid,test}.txt` |
| 3. MLM Fine-Tuning | Specialise ChemBERTa on EGFR chemistry | `scripts/finetune_mlm.py` | `models/chemberta_mlm_ft/` |
| 4. QSAR Oracle | Learn potency scoring function | `scripts/train_qsar_rf.py` | `models/egfr_rf_qsar.pkl` |
| 5. RL Optimisation | Optimise molecule policy with composite reward | `scripts/rl_finetune.py` | `models/chemberta_rl_ft/` |
| 6. Molecule Generation | Sample, deduplicate, validate SMILES | `scripts/generate_molecules.py` | `results/generated_*.txt` |
| 7. 3D Prep | Optional: embed + convert to PDBQT | `scripts/prepare_3d_ligands.py` | `ligands/*/` |
| 8. Docking & Analysis | Score vs. receptor, aggregate metrics | `docking/docking.sh`, `docking/EXTRACT.py`, `scripts/analyze_results.py` | `results/docking_scores.csv`, plots |

## Project Layout
```bash
RL-LLM-GenAI-EGFR-De-Novo-Small-Drug-Molecule/
├── scripts/                # Reproducible CLI entry points
├── data/                   # Cleaned CSV + SMILES splits (mount or copy)
├── models/                 # Prior, MLM, and RL checkpoints
├── docking/                # AutoDock Vina launcher + log parser
├── ligands/                # PDBQT inputs for docking runs
├── results/                # Generated SMILES, docking tables, plots
├── figures/                # Project diagrams / publication assets
└── README.md               # You are here
```

## Getting Started
**Requirements**
- Python 3.9+
- `pip install -r requirements.txt`
- External: AutoDock Vina ≥1.2.7, RDKit, (optional) OpenBabel or MGLTools for ligand prep.

**Environment Bootstrap**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you plan to run offline, download `seyonec/ChemBERTa-zinc-base-v1` ahead of time and point scripts to `models/chemberta_prior/`.

## End-to-End Runbook
1. **Prep the data**
   ```bash
   python scripts/data_download_and_clean.py --out data/all_egfr_ic50_data_cleaned.csv
   python scripts/split_dataset.py \
       --input data/all_egfr_ic50_data_cleaned.csv \
       --smiles_col canonical_smiles --label_col pIC50 \
       --out_prefix data/egfr
   ```
2. **Fine-tune the prior**
   ```bash
   python scripts/finetune_mlm.py \
       --train data/egfr_train.txt --valid data/egfr_valid.txt \
       --out_dir models/chemberta_mlm_ft
   ```
3. **Train the potency oracle**
   ```bash
   python scripts/train_qsar_rf.py \
       --input data/all_egfr_ic50_data_cleaned.csv \
       --smiles_col canonical_smiles --label_col pIC50 \
       --out models/egfr_rf_qsar.pkl
   ```
4. **Reinforcement learning optimisation**
   ```bash
   python scripts/rl_finetune.py \
       --train data/egfr_train.txt --valid data/egfr_valid.txt \
       --rf_model models/egfr_rf_qsar.pkl \
       --out_dir models/chemberta_rl_ft
   ```
5. **Sample molecules & prep for docking**
   ```bash
   python scripts/generate_molecules.py \
       --model models/chemberta_rl_ft --num 1000 \
       --out_smiles results/generated_1000.txt

   python scripts/prepare_3d_ligands.py \
       --input results/generated_1000.txt \
       --out_dir ligands/generated_ligand_3d
   ```
6. **Dock & analyse**
   ```bash
   bash docking/docking.sh
   python docking/EXTRACT.py \
       --dirs ligands/original_ligand_3d ligands/generated_ligand_3d \
       --out results/docking_scores.csv
   python scripts/analyze_results.py \
       --docking_csv results/docking_scores.csv \
       --out_dir results
   ```

## Result Highlights
![Docking Score Distribution](figures/docking_score_distribution.pdf)

The plot is generated directly from `scripts/analyze_results.py` to provide docking score distribution (RL vs Baseline). Additional figures live in `figures/` for slide-ready storytelling.

## Troubleshooting & Tips
- **RDKit installation** varies by OS; follow the official docs linked in `requirements.txt` comments.
- **Docking grid**: update grid box coordinates in `docking/docking.sh` to match your receptor preparation.
- **Checkpoint hygiene**: keep prior, MLM, and RL checkpoints separate — this makes ablations painless.
- **Offline workflow**: populate `models/chemberta_prior/` with a local copy of ChemBERTa to avoid download prompts.

## Roadmap
- [ ] Integrate diffusion-based generative baselines for head-to-head benchmarking.
- [ ] Add GPU-accelerated docking via DiffDock or Gnina.
- [ ] Provide wandb/MLflow logging hooks for long-running experiments.
- [ ] Ship a lightweight Streamlit app for medicinal chemist review.

Contributions via issues or PRs are warmly welcomed — please share what dataset or receptor variants you are targeting.

## Citations
- Ahmad, et al. **ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction** (2022).
- Landrum, G. **RDKit: Open-source cheminformatics** (2025).
- Eberhardt, et al. **AutoDock Vina 1.2.0: New Docking Methodologies** (2021).
- Stamos, et al. **Structure of the EGFR kinase domain** (2002).

---

If this workflow accelerates your EGFR project, consider starring ⭐ the repository and letting us know what breakthroughs you unlock!
