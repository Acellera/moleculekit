# How to select atoms

## Goal

Produce a boolean mask or integer index array identifying a subset of atoms in a {py:class}`~moleculekit.molecule.Molecule`.

## Minimal example

```python
from moleculekit.molecule import Molecule

mol = Molecule("3PTB")

# String selection — returns boolean mask
mask = mol.atomselect("protein and name CA")

# Same selection as integer indices
idx = mol.atomselect("protein and name CA", indexes=True)
```

## Parameters that matter

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `sel` | `str`, `np.ndarray` | required | Selection string, boolean mask, or integer index array |
| `indexes` | `bool` | `False` | Return integer indices instead of a boolean mask |
| `strict` | `bool` | `False` | Raise an error if the selection is empty |
| `fileBonds` | `bool` | `True` | Use bonds loaded from the file for bond-dependent keywords |
| `guessBonds` | `bool` | `True` | Fall back to distance-based bond guessing when needed |

## Common variations

```python
# Build a mask directly from per-atom arrays — faster than re-parsing a string
mask = (mol.chain == "A") & (mol.resid > 100)
```

```python
# A precomputed mask can index per-atom arrays directly
ca_mask = mol.atomselect("name CA")
ca_coords = mol.coords[ca_mask]      # shape (n_CA, 3, numFrames)
ca_names  = mol.name[ca_mask]
```

## Gotchas

- An empty selection returns an empty mask (no error) unless `strict=True`.
- Bond-dependent selectors (`protein`, `nucleic`, `same residue as`, `same fragment as`) require bonds; if `mol.bonds` is empty the engine falls back to guessing. `within` and `exwithin` are distance-only and do **not** depend on bonds.
- Passing a boolean mask or integer array bypasses the string parser entirely, which is significantly faster when reusing the same selection in a loop.
- Precomputed masks become stale after any operation that changes the atom count or order (e.g. {py:meth}`~moleculekit.molecule.Molecule.filter`, {py:meth}`~moleculekit.molecule.Molecule.remove`, {py:meth}`~moleculekit.molecule.Molecule.append`).

## See also

- [Atom-selection language](../explanation/atom-selection-language.md)
- [How to filter and remove atoms](filter-and-remove.md)
- [How to set and get properties](set-and-get-properties.md)
