from functools import cached_property
from typing import Dict, Type, List, Union, Iterable
from pathlib import Path

import numpy as np
import selfies as sf
from ruamel.yaml import YAML

src_path = Path(__file__)
top = src_path.parents[2].absolute()


def get_paths() -> Dict[str, Type[Path]]:
    """
    Retrieve a dictionary containing the absolute paths
    for this project. This provides a simple method for
    traversing and referencing files outside the current
    working directory, particularly for scripts and notebooks.
    """
    paths = {
        "data": top.joinpath("data"),
        "models": top.joinpath("models"),
        "notebooks": top.joinpath("notebooks"),
        "scripts": top.joinpath("scripts"),
    }
    for subfolder in ["raw", "interim", "external", "processed"]:
        paths[subfolder] = paths.get("data").joinpath(subfolder)
    return paths


def get_pretrained_path() -> Type[Path]:
    return src_path.parent.joinpath("models/pretrained")


class Translator(object):
    def __init__(self, alphabet: List[str], max_length: int):
        self.alphabet = alphabet
        self.max_length = max_length

    @property
    def max_length(self) -> int:
        return self._max_length

    @max_length.setter
    def max_length(self, value: int):
        assert value > 0
        self._max_length = value

    @cached_property
    def token_map(self):
        return {s: i for i, s in enumerate(self.alphabet)}

    @property
    def alphabet(self):
        return self._alphabet

    @alphabet.setter
    def alphabet(self, alphabet: List[str]):
        if isinstance(alphabet, list):
            self._alphabet = alphabet

    def __len__(self) -> int:
        return len(self.alphabet)

    def tokenize(self, selfies: str) -> List[int]:
        """
        For backwards compatibility, this tokenizes SELFIES
        for now.
        """
        return self.tokenize_selfies(selfies)

    def tokenize_selfies(self, selfies: str) -> List[int]:
        label, onehot = sf.selfies_to_encoding(selfies, self.token_map, self.max_length)
        return label, onehot

    def tokenize_smiles(self, smiles: str) -> List[int]:
        selfie = sf.encoder(smiles)
        return self.tokenize_selfies(selfie)

    def index_to_character(self, index: int) -> str:
        return self.alphabet[index]

    def indices_to_selfies(self, sentence: Iterable[int]) -> str:
        characters = [
            self.index_to_character(item) for item in sentence if item != "[nop]"
        ]
        return "".join(characters)

    def indices_to_smiles(self, sentence: Iterable[int]) -> str:
        selfie = self.indices_to_selfies(sentence)
        return sf.decoder(selfie)

    @classmethod
    def from_yaml(cls, yaml_path):
        yaml = YAML()
        with open(yaml_path) as read_file:
            data = yaml.load(read_file)
        return cls(data.get("alphabet"), data.get("max_length"))

    @classmethod
    def from_pretrained(cls):
        path = get_pretrained_path().joinpath("translator.yml")
        return cls.from_yaml(path)

    def __repr__(self) -> str:
        return f"Translator with {len(self.alphabet)} tokens, padded to {self.max_length} length."

    def to_yaml(self, yaml_path: str):
        target = Path(yaml_path)
        yaml = YAML()
        with open(target, "w+") as write_file:
            yaml.dump(
                {"alphabet": self.alphabet, "max_length": self.max_length}, write_file
            )
