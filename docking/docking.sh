#!/usr/bin/env bash
set -euo pipefail

# Docking for original small molecules (top 100 pIC50 molecules in the EGFR dataset)
for i in ./ligands/original_ligand_3d/lig*.pdbqt; do
  ./vina_1.2.7_mac_aarch64 \
    --receptor 1M17_clean.pdbqt \
    --ligand "$i" \
    --center_x 21.857 \
    --center_y 0.260 \
    --center_z 52.761 \
    --size_x 20 \
    --size_y 20 \
    --size_z 20 \
    --out "${i%.pdbqt}_out.pdbqt" \
    > "${i%.pdbqt}_out.txt"
done

# Docking for de novo generated small molecules
for i in ./ligands/generated_ligand_3d/lig*.pdbqt; do
  ./vina_1.2.7_mac_aarch64 \
    --receptor 1M17_clean.pdbqt \
    --ligand "$i" \
    --center_x 21.857 \
    --center_y 0.260 \
    --center_z 52.761 \
    --size_x 20 \
    --size_y 20 \
    --size_z 20 \
    --out "${i%.pdbqt}_out.pdbqt" \
    > "${i%.pdbqt}_out.txt"
done

