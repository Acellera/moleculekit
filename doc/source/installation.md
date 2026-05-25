# Installation

MoleculeKit installs cleanly with `pip` or `conda`. We recommend creating an isolated environment regardless of the package manager you use.

## With pip

```bash
pip install moleculekit
```

For an isolated environment, create one with `venv`, `conda`, or another tool first:

```bash
python -m venv .venv
source .venv/bin/activate
pip install moleculekit
```

## With conda

Install Miniforge from <https://github.com/conda-forge/miniforge>, then:

```bash
conda create -n moleculekit
conda activate moleculekit
conda install moleculekit -c acellera -c conda-forge
```

## Verify

```python
from moleculekit.molecule import Molecule
mol = Molecule("3PTB")
print(mol.numAtoms)
```

Expected output: an integer count of atoms in PDB entry `3PTB`.

## For contributors: `uv`

If you are developing moleculekit (running the test suite, building docs, working from a checkout), `uv` is the recommended tool for managing the project environment:

```bash
uv sync --group dev      # install dev + test deps
uv sync --group docs     # add the doc-build deps
uv run pytest            # run the test suite
```

See the `pyproject.toml` `[dependency-groups]` table for the full list of optional dependency groups.
