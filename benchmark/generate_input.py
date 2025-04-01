#!/usr/bin/env python3
"""
Script to process analytical data and generate tokenized datasets for SMILES and spectra.
"""

from pathlib import Path
from typing import Tuple, List, Dict, Union
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import os
import click
import numpy as np
import pandas as pd
import regex as re

from rdkit import Chem
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from rxn.chemutils.tokenization import tokenize_smiles


def set_nice():
    os.nice(10)  # Increase niceness by 10 (lower priority)


def split_data(data: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into train, test, and validation sets.

    Parameters:
        data: Input DataFrame.
        seed: Random seed for reproducibility.

    Returns:
        A tuple containing the train, test, and validation DataFrames.
    """
    train, test = train_test_split(data, test_size=0.1, random_state=seed, shuffle=True)
    train, val = train_test_split(train, test_size=0.05, random_state=seed, shuffle=True)
    return train, test, val


def add_explicit_h(smiles: str) -> str:
    """
    Add explicit hydrogens to a SMILES string.

    Parameters:
        smiles: The SMILES string.

    Returns:
        A SMILES string with explicit hydrogens.
    """
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), allHsExplicit=True)


def tokenize_formula(formula: str) -> str:
    """
    Tokenize a molecular formula into spaced tokens.

    Parameters:
        formula: Molecular formula string.

    Returns:
        A tokenized formula string.
    """
    return ' '.join(re.findall(r"[A-Z][a-z]?|\d+|.", formula)) + ' '


def process_hnmr(multiplets: List[Dict[str, Union[str, float, int]]]) -> str:
    """
    Process 1H NMR multiplets into a tokenized string.

    Parameters:
        multiplets: List of dictionaries containing multiplet information.

    Returns:
        A tokenized 1H NMR string.
    """
    multiplet_str = "1HNMR | "
    for peak in multiplets:
        range_max = float(peak["rangeMax"])
        range_min = float(peak["rangeMin"])
        formatted_peak = "{:.2f} {:.2f} ".format(range_max, range_min)
        formatted_peak += "{} {}H ".format(peak["category"], peak["nH"])
        js = str(peak["j_values"])
        if js != "None":
            split_js = list(filter(None, js.split("_")))
            processed_js = ["{:.2f}".format(float(j)) for j in split_js]
            formatted_peak += "J " + " ".join(processed_js)
        multiplet_str += formatted_peak.strip() + " | "
    return multiplet_str[:-2]


def process_cnmr(carbon_nmr: List[Dict[str, Union[str, float, int]]]) -> str:
    """
    Process 13C NMR peaks into a tokenized string.

    Parameters:
        carbon_nmr: List of dictionaries containing 13C NMR data.

    Returns:
        A tokenized 13C NMR string.
    """
    nmr_string = "13CNMR "
    for peak in carbon_nmr:
        nmr_string += f"{round(float(peak['delta (ppm)']), 1)} "
    return nmr_string


def process_ir(ir: np.ndarray, interpolation_points: int = 400) -> str:
    """
    Interpolate and normalize IR spectra into a tokenized string.

    Parameters:
        ir: Array containing IR spectral intensities.
        interpolation_points: Number of points to interpolate.

    Returns:
        A tokenized IR string.
    """
    original_x = np.linspace(400, 4000, 1800)
    interpolation_x = np.linspace(400, 4000, interpolation_points)
    interp = interp1d(original_x, ir)
    interp_ir = interp(interpolation_x)
    # Normalize
    interp_ir = interp_ir + abs(min(interp_ir))
    interp_ir = (interp_ir / max(interp_ir)) * 100
    interp_ir = np.round(interp_ir, decimals=0).astype(int).astype(str)
    return 'IR ' + ' '.join(interp_ir) + ' '


def process_msms(msms: List[List[float]]) -> str:
    """
    Process MS/MS data into a tokenized string.

    Parameters:
        msms: List of [m/z, intensity] pairs.

    Returns:
        A tokenized MS/MS string.
    """
    msms_string = ""
    for peak in msms:
        msms_string += "{:.1f} {:.1f} ".format(round(peak[0], 1), round(peak[1], 1))
    return msms_string


def tokenise_data(
        *,
        data: pd.DataFrame,
        h_nmr: bool,
        c_nmr: bool,
        ir: bool,
        pos_msms: bool,
        neg_msms: bool,
        formula: bool,
        explicit_h: bool,
        mono: bool,
) -> pd.DataFrame:
    """
    Tokenize the data from the DataFrame into input/target pairs.

    Parameters:
        data: Input DataFrame containing analytical data.
        h_nmr: Whether to include 1H NMR data.
        c_nmr: Whether to include 13C NMR data.
        ir: Whether to include IR spectra.
        pos_msms: Whether to include positive MS/MS data.
        neg_msms: Whether to include negative MS/MS data.
        formula: Whether to include molecular formula.
        explicit_h: Whether to convert SMILES to explicit hydrogen representation.
        mono: Whether to remove stereo information from SMILES.

    Returns:
        A DataFrame with tokenized 'source' and 'target' columns.
    """
    input_list = []

    # for (_, row) in tqdm(data.iterrows(), total=len(data)):
    for (_, row) in data.iterrows():
        tokenized_formula = tokenize_formula(row['molecular_formula'])
        tokenized_input = tokenized_formula if formula else ""

        if h_nmr:
            tokenized_input += process_hnmr(row['h_nmr_peaks'])
        if c_nmr:
            tokenized_input += process_cnmr(row['c_nmr_peaks'])
        if ir:
            tokenized_input += process_ir(row["ir_spectra"])
        if pos_msms:
            pos_msms_string = (
                    "E0Pos " + process_msms(row["msms_positive_10ev"]) +
                    "E1Pos " + process_msms(row["msms_positive_20ev"]) +
                    "E2Pos " + process_msms(row["msms_positive_40ev"])
            )
            tokenized_input += pos_msms_string
        if neg_msms:
            neg_msms_string = (
                    "E0Neg " + process_msms(row["msms_negative_10ev"]) +
                    "E1Neg " + process_msms(row["msms_negative_20ev"]) +
                    "E2Neg " + process_msms(row["msms_negative_40ev"])
            )
            tokenized_input += neg_msms_string

        smiles = row["smiles"]

        if explicit_h:
            smiles = add_explicit_h(smiles)

        if mono:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)

        tokenized_target = tokenize_smiles(smiles=smiles)
        input_list.append({'source': tokenized_input.strip(), 'target': tokenized_target})

    input_df = pd.DataFrame(input_list)

    return input_df.drop_duplicates(subset="source")


def save_set(data_set: pd.DataFrame, out_path: Path, set_type: str, pred_spectra: bool) -> None:
    """
    Save the tokenized dataset to text files.

    Parameters:
        data_set: DataFrame containing the dataset.
        out_path: Output directory as a Path object.
        set_type: A label for the dataset (e.g., 'train', 'test', 'val').
        pred_spectra: Flag indicating whether to swap source and target.
    """
    out_path.mkdir(parents=True, exist_ok=True)
    smiles = data_set["target"].tolist()
    spectra = data_set["source"].tolist()
    src_items = smiles if pred_spectra else spectra
    tgt_items = spectra if pred_spectra else smiles

    with (out_path / f"src-{set_type}.txt").open("w", encoding="utf-8") as f:
        for item in src_items:
            f.write(f"{item}\n")

    with (out_path / f"tgt-{set_type}.txt").open("w", encoding="utf-8") as f:
        for item in tgt_items:
            f.write(f"{item}\n")


def process_parquet_file(
        parquet_file: Path,
        **params,
) -> pd.DataFrame:
    """
    Process a single parquet file and return its tokenized DataFrame.

    Parameters:
        parquet_file: Path to the parquet file.
        h_nmr, c_nmr, ir, pos_msms, neg_msms, formula, explicit_h: Flags for tokenization options.

    Returns:
        A tokenized DataFrame.
    """
    data = pd.read_parquet(parquet_file)
    return tokenise_data(data=data, **params)


@click.command()
@click.option(
    "--analytical_data",
    "-n",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the NMR dataframe",
)
@click.option(
    "--out_path",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path",
)
@click.option("--h_nmr", is_flag=True, default=False, help="Include 1H NMR data")
@click.option("--c_nmr", is_flag=True, default=False, help="Include 13C NMR data")
@click.option("--ir", is_flag=True, default=False, help="Include IR spectra")
@click.option("--pos_msms", is_flag=True, default=False, help="Include positive MS/MS data")
@click.option("--neg_msms", is_flag=True, default=False, help="Include negative MS/MS data")
@click.option("--formula", is_flag=True, default=False, help="Include molecular formula")
@click.option("--explicit_h", is_flag=True, default=False, help="Use SMILES with explicit hydrogens")
@click.option("--mono", is_flag=True, default=False, help="Remove stereo information from SMILES")
@click.option("--pred_spectra", is_flag=True, default=False, help="Predict spectra")
@click.option("--seed", type=int, default=3245, help="Random seed")
def main(
        analytical_data: Path,
        out_path: Path,
        h_nmr: bool,
        c_nmr: bool,
        ir: bool,
        pos_msms: bool,
        neg_msms: bool,
        formula: bool,
        explicit_h: bool,
        mono: bool,
        pred_spectra: bool,
        seed: int,
) -> None:
    """
    Process analytical data and generate tokenized dataset files.

    Reads all parquet files from the analytical_data directory, tokenizes the information,
    splits it into train, test, and validation sets, and saves them to out_path/data.
    """
    print(f"Analytical data: {analytical_data}")
    print(f"Output path: {out_path}")
    print(f"H NMR: {h_nmr}")
    print(f"C NMR: {c_nmr}")
    print(f"IR: {ir}")
    print(f"Positive MSMS: {pos_msms}")
    print(f"Negative MSMS: {neg_msms}")
    print(f"Formula: {formula}")
    print(f"Explicit H: {explicit_h}")
    print(f"Mono (drop stereo): {mono}")
    print(f"Predict spectra: {pred_spectra}")
    print(f"Seed: {seed}")

    parquet_files = sorted(list(analytical_data.glob("*.parquet")))
    process_func = partial(
        process_parquet_file,
        h_nmr=h_nmr,
        c_nmr=c_nmr,
        ir=ir,
        pos_msms=pos_msms,
        neg_msms=neg_msms,
        formula=formula,
        explicit_h=explicit_h,
        mono=mono,
    )

    with ProcessPoolExecutor(max_workers=7, initializer=set_nice) as executor:
        tokenised_data_list = list(tqdm(
            executor.map(process_func, parquet_files),
            total=len(parquet_files),
        ))

    tokenised_data = pd.concat(tokenised_data_list)
    train_set, test_set, val_set = split_data(tokenised_data, seed)

    out_data_path = out_path / "data"

    save_set(test_set, out_data_path, "test", pred_spectra)
    save_set(train_set, out_data_path, "train", pred_spectra)
    save_set(val_set, out_data_path, "val", pred_spectra)


if __name__ == '__main__':
    main()
