from typing import Union

import pandas as pd
import numpy as np
from rdkit import Chem
from astrochem_ml.smiles.isotopes import isotopologues_from_file, generate_all_isos
from joblib import Parallel, delayed

from astrochem_embedding import get_paths


def convert_inchi_to_smiles(inchi: str) -> Union[None, Chem.Mol]:
    """
    Function to exclude badly InChI from the KIDA molecule
    set. There are a few of them.
    """
    mol = Chem.MolFromInchi(inchi)
    if mol:
        return Chem.MolToSmiles(mol)
    return None


paths = get_paths()

# load in molecules from KIDA
kida_df = pd.read_csv(paths.get("raw").joinpath("kida-molecules_05_Jul_2020.csv"))
# drop rows that don't have data
kida_df.dropna(inplace=True)

smiles = []
# filter out bad InChI and convert to SMILES
for inchi in kida_df["InChI"]:
    smiles.append(convert_inchi_to_smiles(inchi))

smiles = list(filter(lambda x: x is not None, smiles))

# write out the valid KIDA SMILES to file
with open(paths.get("interim").joinpath("kida.smi"), "w+") as write_file:
    for smi in smiles:
        write_file.write(f"{smi}\n")

# exhaustively generate every isotopologues without hydrogen
all_isos = []
for smi in smiles:
    if "c" not in smi:
        all_isos.extend(generate_all_isos(smi, explicit_h=True))
all_isos = list(set(all_isos))

with open(paths.get("interim").joinpath("kida_isotopologues.smi"), "w+") as write_file:
    for iso in all_isos:
        write_file.write(f"{iso}\n")
