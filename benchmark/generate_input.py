#!/usr/bin/env python3
"""
Script to process analytical data and generate tokenized datasets for SMILES and spectra.
"""

import os as so
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional

import click
import numpy as np
import pandas as pd
import regex as re
from rdkit import Chem
from rdkit.Chem import StereoSpecified
from rdkit.Chem.rdmolops import FindPotentialStereo
from rxn.chemutils.tokenization import tokenize_smiles
from scipy.interpolate import interp1d
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


def set_nice():
    so.nice(10)  # Increase niceness by 10 (lower priority)


def smiles_for_groups(smiles: str) -> str:
    """
    Normalizes SMILES strings for group-aware splitting in the most permissive way,
    in particular by removing stereochemistry.
    """
    if not (mol := Chem.MolFromSmiles(smiles)):
        raise ValueError(f"Invalid SMILES: {smiles}")

    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True, allHsExplicit=False)


def split_data(data: pd.DataFrame, seed: int, groups: Optional[pd.Series] = None) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a molecular dataset into training, test, and validation sets.

    If group labels are provided, a group-aware strategy is used to ensure that the test set
    contains entire groups (e.g., chemical clusters or scaffolds), avoiding data leakage
    between training and test.

    Key behavior:
    - Exactly 10% of the entire dataset is allocated to the test set.
    - If `groups` is provided:
        * SMILES are normalized for consistency.
        * Group-aware splitting ensures no group appears in both train and test.
        * An assertion guarantees that the group-aware split yields enough samples for the test set.
        * The remaining 90% is split randomly: 95% for training and 5% for validation.
    - If `groups` is not provided:
        * A random split is used: 10% test, then 5% of remaining 90% for validation.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing a 'smiles' column.
        seed (int): Random seed for reproducibility.
        groups (Optional[pd.Series]): Optional Series mapping SMILES strings to group IDs,
                                       used for group-aware splitting. Index should be SMILES.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing (train_df, test_df, val_df),
        with 'smiles' and all original columns preserved.
    """

    data = data.copy()

    if groups is not None:
        # Normalize SMILES
        data['normalized_smiles'] = data['smiles'].apply(smiles_for_groups)
        groups.index = groups.index.map(smiles_for_groups)

        # Map group IDs
        data['group_id'] = data['normalized_smiles'].map(groups)

        # Use group-aware split for test set
        known = data[data['group_id'].notna()]
        unknown = data[data['group_id'].isna()]

        n_test = int(len(data) * 0.1)

        gss = GroupShuffleSplit(n_splits=1, test_size=n_test / len(known), random_state=seed)
        group_train_idx, group_test_idx = next(gss.split(known, groups=known['group_id']))
        group_test = known.iloc[group_test_idx]

        # Assert test size is as expected
        assert len(group_test) >= n_test, "Group-based test split did not yield enough samples."

        # Remaining data = group_train + all unknowns
        group_train = known.iloc[group_train_idx]
        remaining = pd.concat([group_train, unknown], ignore_index=True)

        train_df, val_df = train_test_split(remaining, test_size=0.05, random_state=seed, shuffle=True)

        drop_cols = ['normalized_smiles', 'group_id']

        return (
            train_df.drop(columns=drop_cols),
            group_test.drop(columns=drop_cols),
            val_df.drop(columns=drop_cols),
        )

    else:
        # No group info: simple random split
        test_df = data.sample(frac=0.1, random_state=seed)
        remaining = data.drop(index=test_df.index)
        train_df, val_df = train_test_split(remaining, test_size=0.05, random_state=seed, shuffle=True)
        return train_df, test_df, val_df


def get_smiles_groups_from_uspto(uspto_path: Path) -> pd.Series:
    return pd.read_csv(uspto_path, sep='\t').set_index('smiles').component


def has_complete_stereo(smiles: str) -> Optional[bool]:
    """
    Check if a SMILES string has complete stereo information.

    Parameters:
        smiles: The SMILES string.

    Returns:
        True if the SMILES has complete stereo information, False otherwise.
        None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol:
        pot_stereo = FindPotentialStereo(mol)
        return all((s.specified == StereoSpecified.Specified) for s in pot_stereo)


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
    interp_ir = (interp_ir / (max(interp_ir) or 1)) * 100
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
        req_stereo: bool,
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
        req_stereo: Whether to require complete stereo information in SMILES.
        explicit_h: Whether to convert SMILES to explicit hydrogen representation.
        mono: Whether to remove all stereo information from SMILES.

    Returns:
        A DataFrame with tokenized 'source' and 'target' columns.
    """
    input_list = []

    # for (_, row) in tqdm(data.iterrows(), total=len(data)):
    for (_, row) in data.iterrows():
        input_smiles = row["smiles"]

        if not Chem.MolFromSmiles(input_smiles):
            print(f"Skipping {input_smiles} due to invalid SMILES")
            continue

        if req_stereo:
            if not has_complete_stereo(input_smiles):
                print(f"Skipping {row['smiles']} due to incomplete stereochemistry")
                continue

        tokenized_input = tokenize_formula(row['molecular_formula']) if formula else ""

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

        smiles = input_smiles

        # The order is important here:

        if mono:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)

        if explicit_h:
            smiles = add_explicit_h(smiles)

        tokenized_target = tokenize_smiles(smiles=smiles)

        input_list.append({'source': tokenized_input.strip(), 'target': tokenized_target, 'smiles': input_smiles})

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
    "--uspto_path",
    type=click.Path(exists=True, path_type=Path),
    required=False,
    help="Path to the USPTO smiles-components dataframe",
    default=None,
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
@click.option("--explicit_h", is_flag=True, default=False, help="Add explicit hydrogens in target SMILES")
@click.option("--req_stereo", is_flag=True, default=False, help="Require complete stereo in source SMILES")
@click.option("--mono", is_flag=True, default=False, help="Remove stereo information from target SMILES")
@click.option("--pred_spectra", is_flag=True, default=False, help="Predict spectra")
@click.option("--seed", type=int, default=3245, help="Random seed")
def main(
        analytical_data: Path,
        uspto_path: Optional[Path],
        out_path: Path,
        h_nmr: bool,
        c_nmr: bool,
        ir: bool,
        pos_msms: bool,
        neg_msms: bool,
        formula: bool,
        explicit_h: bool,
        req_stereo: bool,
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
    print(f"USPTO data: {uspto_path}")
    print(f"Output path: {out_path}")
    print(f"H NMR: {h_nmr}")
    print(f"C NMR: {c_nmr}")
    print(f"IR: {ir}")
    print(f"Positive MSMS: {pos_msms}")
    print(f"Negative MSMS: {neg_msms}")
    print(f"Formula: {formula}")
    print(f"Explicit H: {explicit_h}")
    print(f"Require stereo: {req_stereo}")
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
        req_stereo=req_stereo,
        explicit_h=explicit_h,
        mono=mono,
    )

    with ProcessPoolExecutor(max_workers=7, initializer=set_nice) as executor:
        tokenised_data_list = list(tqdm(
            executor.map(process_func, parquet_files),
            total=len(parquet_files),
        ))

    groups: Optional[pd.Series] = get_smiles_groups_from_uspto(uspto_path) or None

    tokenised_data = pd.concat(tokenised_data_list)
    (train_set, test_set, val_set) = split_data(tokenised_data, seed=seed, groups=groups)

    out_data_path = out_path / "data"

    save_set(test_set, out_data_path, "test", pred_spectra)
    save_set(train_set, out_data_path, "train", pred_spectra)
    save_set(val_set, out_data_path, "val", pred_spectra)


if __name__ == '__main__':
    main()
