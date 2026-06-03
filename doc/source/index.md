# MoleculeKit

A Python library for reading, writing, and manipulating biomolecular structures and trajectories. Built around the {py:class}`~moleculekit.molecule.Molecule` class for proteins, nucleic acids, and MD systems, with a full system-preparation pipeline ({py:func}`~moleculekit.tools.preparation.systemPrepare`) that handles protonation at a chosen pH, non-standard residues, custom residues from SMILES, mutation, and gap modeling.

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🎓 Tutorials
:link: tutorials/index
:link-type: doc

Step-by-step lessons. Start here if you're new.
:::

:::{grid-item-card} 🛠 How-to guides
:link: how-to/index
:link-type: doc

Task-focused recipes. "How do I X?"
:::

:::{grid-item-card} 📖 Reference
:link: reference/index
:link-type: doc

Full API documentation, generated from source.
:::

:::{grid-item-card} 💡 Explanation
:link: explanation/index
:link-type: doc

Concepts and mental models.
:::

::::

## Installation

```bash
pip install moleculekit
```

See [Installation](installation.md) for conda, `uv`, and the full setup.

## Quick start

```python
from moleculekit.molecule import Molecule
from moleculekit.tools.preparation import systemPrepare

mol = Molecule("3PTB")   # fetches trypsin from RCSB

# Template the benzamidine ligand (BEN) from its RCSB SMILES so the
# preparation pipeline gets correct bond orders, formal charges, and
# hydrogens for the non-canonical residue.
mol.templateResidueFromSmiles(
    mol.resname == "BEN",
    smiles="NC(=N)c1ccccc1",
    addHs=True,
)

# Add hydrogens to the whole system, predict pKa values, and titrate
# at the chosen pH.
prepared, specs = systemPrepare(mol, pH=7.4)

# Open the prepared structure in the built-in browser viewer.
prepared.view(viewer="molstar")
```

## Citing

Stefan Doerr, Matthew J. Harvey, Frank Noé, and Gianni De Fabritiis.
*HTMD: High-throughput molecular dynamics for molecular discovery.*
J. Chem. Theory Comput. **2016**, 12 (4), 1845–1852.
[doi:10.1021/acs.jctc.6b00049](http://pubs.acs.org/doi/abs/10.1021/acs.jctc.6b00049)

```{toctree}
:maxdepth: 1
:hidden:

installation
tutorials/index
how-to/index
explanation/index
reference/index
```
