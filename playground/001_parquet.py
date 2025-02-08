import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# file = Path('.').absolute() / "data/multimodal_spectroscopic_dataset/aligned_chunk_0.parquet"
file = Path('.').absolute().parent / "data/multimodal_spectroscopic_dataset/aligned_chunk_0.parquet"

df = pd.read_parquet(str(file), engine="pyarrow")

print(df.head())


first = next(df.iterrows())[1]
print(first)

h_nmr_peaks = first['h_nmr_peaks']
h_nmr_spectra = first['h_nmr_spectra']

h_nmr_spectra.shape


# Assuming h_nmr_spectra is given as a NumPy array of shape (10000,)
# It represents intensity values, with ppm values ranging from 0 to 10.

# Generate ppm scale based on given shape
num_points = h_nmr_spectra.shape[0]
ppm_values = np.linspace(0, 10, num_points)  # Chemical shift (ppm)

# Create the NMR plot
plt.figure(figsize=(10, 5))
plt.plot(ppm_values, h_nmr_spectra, color="black", linewidth=1)
plt.gca().invert_xaxis()  # Invert x-axis for NMR convention
plt.xlabel("Chemical Shift (ppm)")
plt.ylabel("Intensity")
plt.title("¹H-NMR Spectrum")
plt.grid(True)

# Show the plot
plt.show()
plt.pause(1)
