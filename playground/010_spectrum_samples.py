import os
import pandas as pd
import numpy as np

from pathlib import Path
from plox import Plox

from rdkit import Chem
from rdkit.Chem import Draw

OUT_DIR = Path(__file__).with_suffix('')
OUT_DIR.mkdir(exist_ok=True)
print(f"{OUT_DIR = }")

[dataset_path] = (Path(os.environ['HOME']) / "Datasets").glob("**/multimodal_spectroscopic_dataset")

[parquet_file] = dataset_path.glob("aligned_chunk_0.parquet")
print(parquet_file)

df = pd.read_parquet(str(parquet_file), engine="pyarrow")

# First row:
data: pd.Series
#print(f"{data.index = }")

for (i, data) in df.iterrows():
    if i >= 1:
        if len(data['smiles']) >= 10:
            continue

    print(f"SMILES: {data['smiles']}")

    row_dir = OUT_DIR / f"{parquet_file.name}.row-{i}"
    row_dir.mkdir(exist_ok=True)

    for c in ['h_nmr_spectra', 'h_nmr_peaks', 'c_nmr_spectra', 'c_nmr_peaks', 'smiles']:
        with (row_dir / f"{c}.txt").open('w') as f:
            try:
                data[c].tofile(f, sep='\n')
            except AttributeError:
                print(data[c], file=f)

    for c in ['h_nmr_spectra', 'c_nmr_spectra']:
        with Plox() as px:
            # Zoom in on the part that has peaks (strip zeros)
            spectrum = data[c]
            idx = np.where(spectrum != 0)
            (a, b) = (np.min(idx), np.max(idx))
            spectrum = spectrum[a:(b + 1)]

            px.a.plot(np.arange(a, b + 1), spectrum, label=c, lw=0.5)
            px.a.set_xlabel("Index")
            px.a.set_ylabel("Intensity")
            px.a.legend()

            title = {'h_nmr_spectra': "¹H-NMR", 'c_nmr_spectra': "¹³C-NMR"}[c]
            px.a.set_title(f"{title} spectrum of\n{data['smiles']}")
            px.f.savefig(row_dir / f"{c}.png", dpi=600)

    # Plot the molecular structure
    smiles = data['smiles']
    mol = Chem.MolFromSmiles(smiles)
    Draw.MolToFile(mol, row_dir / f"molecular.png", size=(300, 300))
