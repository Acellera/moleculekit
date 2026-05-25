---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# First molecule

**You will learn:** how to fetch a structure from the PDB, inspect its contents, filter atoms, and write it back to disk.

**Prerequisites:**
- moleculekit installed.

## Setup

Import {py:class}`~moleculekit.molecule.Molecule` — the central class for all structure manipulation in moleculekit.

```{code-cell} python
from moleculekit.molecule import Molecule
```

## Step 1 — Load a structure

```{code-cell} python
mol = Molecule("3PTB")
```

The {py:class}`~moleculekit.molecule.Molecule` constructor accepts either a
local file path (PDB, mmCIF, MOL2, PRMTOP, PSF, ... — see [How to read a
structure](../howto/read-a-structure.md) for the full list of supported
formats) or a four-character RCSB PDB ID, which it downloads and parses
on the fly. Here we use the PDB ID `3PTB`: bovine trypsin, 1701 atoms
covering one protein chain, a shell of crystallographic water molecules,
a calcium ion, and the benzamidine ligand in the active site.

## Step 2 — Inspect basics

```{code-cell} python
mol.numAtoms
```

`numAtoms` is a single integer — the total number of atoms in the loaded structure.

```{code-cell} python
mol.numFrames
```

`numFrames` is 1 for a static PDB; it grows when you load a trajectory.

```{code-cell} python
sorted(set(mol.resname))
```

`mol.resname` is an array with one entry per atom; wrapping it in `set` gives
the unique residue names present.  You should see standard amino-acid codes
alongside `BEN` (benzamidine), `CA` (calcium), and `HOH` (water).

## Step 3 — Inspect per-atom properties

Every per-atom field on a `Molecule` is a NumPy array of length `mol.numAtoms`.
The arrays are indexed in parallel — `mol.name[i]`, `mol.resname[i]`,
`mol.resid[i]`, and `mol.chain[i]` all describe the same atom.

```{code-cell} python
mol.name[:8]
```

Atom names as they appear in the source file (`N`, `CA`, `C`, `O`, ... for
the protein backbone).

```{code-cell} python
mol.element[:8]
```

Element symbols per atom.

```{code-cell} python
sorted(set(mol.chain)), sorted(set(mol.segid))
```

`chain` (one character, PDB convention) and `segid` (up to four characters,
MD topology convention) are both per-atom arrays. The BCIF fetch of 3PTB
populates `segid` with `['1', '2', '3', '4']` for the deposited entities;
a plain PDB load typically leaves it empty. See [Assign segments and chains](../howto/assign-segments-and-chains.md)
if you need to populate `segid` for an MD parameterization tool.

```{code-cell} python
mol.coords.shape
```

Coordinates are stored as a single `(numAtoms, 3, numFrames)` array. For
this static PDB the third dimension is 1.

Because every field is a NumPy array, the usual NumPy operations work
directly — masking, slicing, `np.unique`, comparisons, and so on. See
[The Molecule data model](../explanation/molecule-data-model.md) for the
full per-atom field list and their dtypes.

## Step 4 — Filter waters

```{code-cell} python
mol.filter("not water")
mol.numAtoms
```

{py:meth}`~moleculekit.molecule.Molecule.filter` mutates `mol` in place,
keeping only the atoms that match the selection string.  This contrasts
with {py:meth}`~moleculekit.molecule.Molecule.remove`, which takes atoms
*out* by matching them.  After dropping the crystallographic waters you
have 1639 atoms remaining (1701 − 62 water oxygens).

## Step 5 — Write the prepared structure

```{code-cell} python
mol.write("trypsin_dry.cif")
```

{py:meth}`~moleculekit.molecule.Molecule.write` infers the format from the
file extension.  The output is written to the current working directory
unless you pass a full path.

## Recap

- Load a structure by PDB ID: `Molecule("3PTB")` fetches it from RCSB automatically.
- Inspect counts and contents with `numAtoms`, `numFrames`, and array attributes such as `resname`, `chain`, and `segid`.
- {py:meth}`~moleculekit.molecule.Molecule.filter` mutates the molecule in place to keep only the atoms you need; then {py:meth}`~moleculekit.molecule.Molecule.write` saves it in any supported format.

## Next

- [Read a structure from a local file](../howto/read-a-structure.md)
- [Molecule data model](../explanation/molecule-data-model.md)
