#!/bin/bash
# RA, 2025-03-06

set -e

CONDA="${HOME}/Programs/miniconda3/"

# Toy dataset (for testing)
DATA="./data/example_data"
HNMR_SPLIT_DATASET_DIR="./runs/000_tiny/h_nmr"

USPTO_PATH=$(ls -t $HOME/repos/nmr-msc/code/data/2024-Alberts/06_uspto/a_smiles_uspto_graph_components/*/smile_to_component.tsv.gz | head -n1)

source "${CONDA}/etc/profile.d/conda.sh"
conda activate fork-of-multimodal-spectroscopic-dataset

python ./benchmark/generate_input.py \
      --analytical_data "${DATA}" \
      --out_path "${HNMR_SPLIT_DATASET_DIR}" \
      --uspto_path "${USPTO_PATH}" \
      --formula \
      --explicit_h \
      --req_stereo \
      --mono \
      --h_nmr

# Zip the split dataset
zip -r -9 "${HNMR_SPLIT_DATASET_DIR}.zip" "${HNMR_SPLIT_DATASET_DIR}"
