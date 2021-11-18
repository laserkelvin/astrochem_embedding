from joblib import Parallel, delayed
from selfies import encoder
from rdkit import Chem

from astrochem_embedding import get_paths


def error_selfies(smi: str):
    try:
        return encoder(smi)
    except:
        return None


def conditional(smi: str) -> bool:
    return all([Chem.MolFromSmiles(smi), "." not in smi])


paths = get_paths()

all_smiles = []

files = [
    paths.get("raw").joinpath("collected_smiles.smi"),
    paths.get("interim").joinpath("kida_isotopologues.smi"),
]

for f in files:
    with open(f) as read_file:
        all_smiles.extend(read_file.readlines())

# get only unique entries
all_smiles = list(set(all_smiles))
all_smiles = [smi for smi in all_smiles if conditional(smi)]

# remove stuff
all_smiles = [smi.strip() for smi in all_smiles]

selfies = Parallel(n_jobs=16)(delayed(error_selfies)(smi) for smi in all_smiles)

with open(paths.get("processed").joinpath("selfies.txt"), "w+") as write_file:
    for selfie in selfies:
        if selfie:
            write_file.write(f"{selfie}\n")
