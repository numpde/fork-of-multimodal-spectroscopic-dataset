#!/bin/bash
# RA, 2025-03-06

set -e

CONDA="${HOME}/Programs/miniconda3/"

DATA="${HOME}/Datasets/2024-Alberts/b_extracted/multimodal_spectroscopic_dataset/"
HNMR_SPLIT_DATASET_DIR="./runs/runs_f_groups/h_nmr"

# Toy dataset (for testing)
#DATA="./data/example_data"
#HNMR_SPLIT_DATASET_DIR="./runs/example/h_nmr"

source "${CONDA}/etc/profile.d/conda.sh"
conda activate fork-of-multimodal-spectroscopic-dataset

python ./benchmark/generate_input.py \
      --analytical_data "${DATA}" \
      --out_path "${HNMR_SPLIT_DATASET_DIR}" \
      --formula \
      --h_nmr

# Zip the split dataset
zip -r -9 "${HNMR_SPLIT_DATASET_DIR}.zip" "${HNMR_SPLIT_DATASET_DIR}"
