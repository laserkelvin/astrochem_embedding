
import pandas as pd
from rdkit import Chem
from astrochem_ml.smiles.isotopes import isotopologues_from_file
from joblib import Parallel, delayed

from astrochem_embedding import get_paths


def convert_inchi_to_smiles(inchi: str):
    mol = Chem.MolFromInchi(inchi)
    if mol:
        return Chem.MolToSmiles(mol)
    return None


paths = get_paths()

kida_df = pd.read_csv(paths.get("raw").joinpath("kida-molecules_05_Jul_2020.csv"))
kida_df.dropna(inplace=True)

smiles = []
for inchi in kida_df["InChI"]:
    smiles.append(convert_inchi_to_smiles(inchi))

with open(paths.get("interim").joinpath("kida.smi"), "w+") as write_file:
    for smi in smiles:
        if smi:
            write_file.write(f"{smi}\n")

all_isos = isotopologues_from_file(paths.get("interim").joinpath("kida.smi"), 8, explicit_h=True)

with open(paths.get("interim").joinpath("kida_isotopologues.smi"), "w+") as write_file:
    for iso in all_isos:
        write_file.write(f"{iso}\n")
