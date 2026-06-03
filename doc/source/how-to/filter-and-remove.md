# How to filter and remove atoms

## Goal

Drop atoms from a {py:class}`~moleculekit.molecule.Molecule` in place — either keep a subset ({py:meth}`~moleculekit.molecule.Molecule.filter`) or discard a subset ({py:meth}`~moleculekit.molecule.Molecule.remove`).

## Minimal example

```python
from moleculekit.molecule import Molecule

mol = Molecule("3PTB")

# Keep only protein atoms (discards water, ligand, etc.)
mol.filter("protein")

# Alternatively, explicitly remove water
mol.remove("water")
```

## Parameters that matter

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `selection` | `str`, boolean `np.ndarray`, or integer `np.ndarray` | required | Atoms to keep (`filter`) or atoms to remove (`remove`) |

## Common variations

```python
# Use a precomputed boolean mask with filter
protein_mask = mol.atomselect("protein")
mol.filter(protein_mask)
```

```python
# Remove a residue range by resid
mol.remove("resid 200 to 300")
```

## Gotchas

- {py:meth}`~moleculekit.molecule.Molecule.filter` **keeps** the selection; {py:meth}`~moleculekit.molecule.Molecule.remove` **drops** it — the semantics are opposite, which is a common source of confusion.
- Both methods mutate the molecule in place and update `mol.bonds` to reflect the new atom indices.
- After either call, any precomputed boolean masks or index arrays are stale and must be recomputed.
- Both methods accept the full string / boolean-mask / integer-index trio (each input is normalized through {py:meth}`~moleculekit.molecule.Molecule.atomselect`).

## See also

- [How to select atoms](select-atoms.md)
- [How to append and merge molecules](append-and-merge-molecules.md)
