import pandas as pd
import numpy as np

from pathlib import Path
from plox import Plox

from rdkit import Chem
from rdkit.Chem import Draw

OUT_DIR = Path(__file__).with_suffix('')
OUT_DIR.mkdir(exist_ok=True)
print(f"{OUT_DIR = }")

[src_file] = Path('.').absolute().parent.glob("**/h_nmr/data/src-train.txt")
[tgt_file] = Path('.').absolute().parent.glob("**/h_nmr/data/tgt-train.txt")

# Read the first line of the file, e.g.:
# 1HNMR 9.01 8.97 d 1H J 2.00 | 8.67 8.62 dd 1H J 1.65 4.79 | 8.19 8.14 dt 1H J 1.84 8.43 | 7.51 7.45 dd 1H J 4.79 8.45 | 3.18 3.12 t 2H J 5.76 | 1.81 1.71 qt 2H J 5.71 6.92 | 1.09 1.03 t 3H J 7.03
with src_file.open('r') as f:
    peaks: str
    peaks = f.readline().lstrip("1HNMR").strip()
    peaks = peaks.split(' | ')

print("Peaks:", *peaks, sep='\n')

# Read the first line of the file
with tgt_file.open('r') as f:
    smiles = "".join(f.readline().strip().split(' '))

print(f"{smiles = }")

mol = Chem.MolFromSmiles(smiles)

# Draw the molecule to file
Draw.MolToFile(mol, OUT_DIR / f"{smiles}.png", size=(300, 300))
