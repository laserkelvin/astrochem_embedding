from astrochem_embedding import utils


def test_pretrained_path():
    path = utils.get_pretrained_path()
    # make sure model weights are there
    assert path.joinpath("VICGAE.pkl").exists()


def test_translator_pretrained():
    t = utils.Translator.from_pretrained()


def test_translator_features():
    t = utils.Translator.from_pretrained()
    assert t.max_length == 886
    assert t.index_to_character(624) == "[nop]"
    # check the tokenization for benzene
    label, onehot = t.tokenize_smiles("c1ccccc1")
    label = list(filter(lambda x: x != 624, label))
    assert label == [375, 253, 375, 253, 375, 253, 528, 246]
