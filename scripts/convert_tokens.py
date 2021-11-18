import selfies as sf
import numpy as np
from tqdm.auto import tqdm
from ruamel.yaml import YAML
from joblib import Parallel, delayed

from astrochem_embedding import get_paths, Translator


NUM_JOBS = 24


def get_label(selfie: str, translator, index: int, output_array: np.ndarray):
    label, _ = translator.tokenize(selfie)
    output_array[index, :] = label


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


paths = get_paths()

with open(paths.get("processed").joinpath("selfies.txt")) as read_file:
    selfs = list(map(lambda x: x.strip(), read_file.readlines()))
# get the max length for the entire dataset
lengths = Parallel(n_jobs=NUM_JOBS)(delayed(sf.len_selfies)(s) for s in tqdm(selfs))
max_length = max(lengths)
alphabet = sf.get_alphabet_from_selfies(selfs)
alphabet.add("[nop]")
alphabet.add("[unk]")
alphabet = list(sorted(alphabet))

translator = Translator(alphabet, max_length)
print(translator)
translator.to_yaml(paths.get("processed").joinpath("translator.yml"))

vocab_mapping = {s: i for i, s in enumerate(alphabet)}

labels = np.memmap(
    paths.get("processed").joinpath("labels.npy"),
    dtype=np.uint16,
    shape=(len(selfs), max_length),
    mode="w+",
)

Parallel(n_jobs=NUM_JOBS)(
    delayed(get_label)(s, translator, i, labels) for i, s in tqdm(enumerate(selfs))
)
blah = labels.copy()
labels._mmap.close()

np.save(paths.get("processed").joinpath("labels.npy"), blah)
