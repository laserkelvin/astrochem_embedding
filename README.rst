Language models for astrochemistry
==================================

|PyPI| |Status| |Python Version| |License|

|Read the Docs| |Tests| |Codecov|

|pre-commit| |Black|

.. |PyPI| image:: https://img.shields.io/pypi/v/astrochem_embedding.svg
   :target: https://pypi.org/project/astrochem_embedding/
   :alt: PyPI
.. |Status| image:: https://img.shields.io/pypi/status/astrochem_embedding.svg
   :target: https://pypi.org/project/astrochem_embedding/
   :alt: Status
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/astrochem_embedding
   :target: https://pypi.org/project/astrochem_embedding
   :alt: Python Version
.. |License| image:: https://img.shields.io/pypi/l/astrochem_embedding
   :target: https://opensource.org/licenses/MIT
   :alt: License
.. |Read the Docs| image:: https://img.shields.io/readthedocs/astrochem_embedding/latest.svg?label=Read%20the%20Docs
   :target: https://astrochem_embedding.readthedocs.io/
   :alt: Read the documentation at https://astrochem_embedding.readthedocs.io/
.. |Tests| image:: https://github.com/laserkelvin/astrochem_embedding/workflows/Tests/badge.svg
   :target: https://github.com/laserkelvin/astrochem_embedding/actions?workflow=Tests
   :alt: Tests
.. |Codecov| image:: https://codecov.io/gh/laserkelvin/astrochem_embedding/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/laserkelvin/astrochem_embedding
   :alt: Codecov
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black


Features
--------

The goal of this project is to provide off the shelf language models that work
for studies in astrochemistry; the needs for general molecule discovery/chemistry
are different from astrochemistry, such as the emphasis on transient (e.g. open-shell)
molecules and isotopologues.

To support these aspects, we provide here light-weight language models (currently just
a GRU seq2seq model) based off of `SELFIES`_ syntax and PyTorch. Elements of
this project are designed to strike a balance between research agility and use for
production, and a lot of emphasis is placed on reproducibility using PyTorch Lightning
and a general user interface that doesn't force the user to know how to develop neural networks.

The current highlight of this package is the ``VICGAE``, or variance-invariance-covariance
regularized GRU autoencoder (I guess probably ``VICGRUAE`` is more accurate?). I intend to
write this up in a more detailed form in the near future, but the basic premise is this:
a pair of GRUs form a seq2seq model, whose task is to complete SELFIES strings based off
of randomly masked tokens within the molecule. To improve chemical representation learning,
the VIC regularization uses self-supervision to ensure the token embeddings are chemically
descriptive: we encourage variance (e.g. [CH2] is different from [OH]), invariance (e.g. 
isotopic substitution should give more or less the same molecule), and covariance (i.e.
minimizing information sharing between embedding dimensions). While the GRU does the actual
SELFIES reconstruction, the VIC regularization is done at the token embedding level.

This has been tested on a few simple comparisons with cosine similarity, comparing isotopic
substitution, element substitution (i.e. C/Si/Ge), and functional group replacement; things
seem to work well for these simple cases.


Requirements
------------

This package requires Python 3.8+, as it uses some decorators only available after 3.7.


Installation
------------

The simplest way to get ``astrochem_embedding`` is through PyPI:

.. code:: console
    
    $ pip install astrochem_embedding

If you're interested in development, want to train your own model,
or make sure you can take advantage of GPU acceleration, I recommend
using ``conda`` for your environment specification:

.. code:: console

   $ conda create -n astrochem_embedding python=3.8
   $ conda activate astrochem_embedding
   $ pip install poetry
   $ poetry install
   $ conda install -c pytorch torch torchvision cudatoolkit=11.3

Usage
-----

The quickest way to get started is by loading a pre-trained model:

.. code:: python

    >>> from astrochem_embedding import VICGAE
    >>> import torch
    >>> model = VICGAE.from_pretrained()
    >>> model.embed_smiles("c1ccccc1")

will return a `torch.Tensor`. For now the general interface doesn't
support batching SMILES just yet, and so to operate on many SMILES
strings would simply require looping:

.. code:: python

    >>> smiles = ["c1ccccc1", "[C]#N", "[13c]1ccccc1"]
    >>> embeddings = torch.stack([model.embed_smiles(s) for s in smiles])
    # optionally convert back to NumPy arrays
    >>> numpy_embeddings = embeddings.numpy()


Project Structure
-----------------

The project filestructure is laid out as such::

   ├── CITATION.cff
   ├── codecov.yml
   ├── CODE_OF_CONDUCT.rst
   ├── CONTRIBUTING.rst
   ├── data
   │   ├── external
   │   ├── interim
   │   ├── processed
   │   └── raw
   ├── docs
   │   ├── codeofconduct.rst
   │   ├── conf.py
   │   ├── contributing.rst
   │   ├── index.rst
   │   ├── license.rst
   │   ├── reference.rst
   │   ├── requirements.txt
   │   └── usage.rst
   ├── environment.yml
   ├── models
   ├── notebooks
   │   ├── dev
   │   ├── exploratory
   │   └── reports
   ├── noxfile.py
   ├── poetry.lock
   ├── pyproject.toml
   ├── README.rst
   ├── scripts
   │   └── train.py
   └── src
      └── astrochem_embedding
         ├── __init__.py
         ├── layers
         │   ├── __init__.py
         │   ├── layers.py
         │   └── tests
         │       ├── __init__.py
         │       └── test_layers.py
         ├── __main__.py
         ├── models
         │   ├── __init__.py
         │   ├── models.py
         │   └── tests
         │       ├── __init__.py
         │       └── test_models.py
         ├── pipeline
         │   ├── data.py
         │   ├── __init__.py
         │   ├── tests
         │   │   ├── __init__.py
         │   │   ├── test_data.py
         │   │   └── test_transforms.py
         │   └── transforms.py
         └── utils.py

A brief summary of what each folder is designed for:

#. `data` contains copies of the data used for this project. It is recommended to form a pipeline whereby the `raw` data is preprocessed, serialized to `interim`, and when ready for analysis, placed into `processed`.
#. `models` contains serialized weights intended for distribution, and/or testing.
#. `notebooks` contains three subfolders: `dev` is for notebook based development, `exploratory` for data exploration, and `reports` for making figures and visualizations for writeup.
#. `scripts` contains files that meant for headless routines, generally those with long compute times such as model training and data cleaning.
#. `src/astrochem_embedding` contains the common code base for this project.


Code development
----------------

All of the code used for this project should be contained in `src/astrochem_embedding`,
at least in terms of the high-level functionality (i.e. not scripts), and is intended to be
a standalone Python package.

The package is structured to match the abstractions for deep learning, specifically PyTorch, 
PyTorch Lightning, and Weights and Biases, by separating parts of data structures and processing
and model/layer development.

Some concise tenets for development

* Write unit tests as you go.
* Commit changes, and commit frequently. Write `semantic`_ git commits!
* Formatting is done with ``black``; don't fuss about it 😃
* For new Python dependencies, use `poetry add <package>`.
* For new environment dependencies, use `conda env export -f environment.yml`.

Notes on best practices, particularly regarding CI/CD, can be found in the extensive
documentation for the `Hypermodern Python Cookiecutter`_ repository.

License
-------

Distributed under the terms of the `MIT license`_,
*Language models for astrochemistry* is free and open source software.


Issues
------

If you encounter any problems,
please `file an issue`_ along with a detailed description.


Credits
-------

This project was generated from `@laserkelvin`_'s PyTorch Project Cookiecutter, 
a fork of  `@cjolowicz`_'s `Hypermodern Python Cookiecutter`_ template.

.. _@cjolowicz: https://github.com/cjolowicz
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _MIT license: https://opensource.org/licenses/MIT
.. _PyPI: https://pypi.org/
.. _Hypermodern Python Cookiecutter: https://github.com/cjolowicz/cookiecutter-hypermodern-python
.. _file an issue: https://github.com/laserkelvin/astrochem_embedding/issues
.. _pip: https://pip.pypa.io/
.. github-only
.. _Contributor Guide: CONTRIBUTING.rst
.. _Usage: https://astrochem_embedding.readthedocs.io/en/latest/usage.html
.. _semantic: https://gist.github.com/joshbuchea/6f47e86d2510bce28f8e7f42ae84c716
.. _@laserkelvin: https://github.com/laserkelvin
.. _SELFIES: https://github.com/aspuru-guzik-group/selfies
