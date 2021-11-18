from typing import Union

import pandas as pd
from rdkit import Chem
from astrochem_ml.smiles.isotopes import isotopologues_from_file
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

# write out the valid KIDA SMILES to file
with open(paths.get("interim").joinpath("kida.smi"), "w+") as write_file:
    for smi in smiles:
        if smi:
            write_file.write(f"{smi}\n")

# exhaustively generate every isotopic combination to
# the abundance of deuterium
all_isos = isotopologues_from_file(
    paths.get("interim").joinpath("kida.smi"), 24, explicit_h=True
)

with open(paths.get("interim").joinpath("kida_isotopologues.smi"), "w+") as write_file:
    for iso in all_isos:
        write_file.write(f"{iso}\n")
