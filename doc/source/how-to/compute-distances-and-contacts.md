# How to compute distances and contacts

## Goal

Compute pairwise distances between atom sets or detect steric clashes between selections.

## Minimal example

```python
from moleculekit.molecule import Molecule
from moleculekit.distance import cdist

mol = Molecule("3PTB")

# Indices for two groups
idx1 = mol.atomselect("protein and name CA", indexes=True)
idx2 = mol.atomselect("resname BEN", indexes=True)

# Pairwise distances between the two groups for frame 0
dists = cdist(mol.coords[idx1, :, 0], mol.coords[idx2, :, 0])
print(dists.shape)  # (n_CA, n_BEN)
```

## Parameters that matter

| Function | Arguments | What it does |
|---|---|---|
| `cdist(coords1, coords2)` | Two `(N, 3)` arrays | All pairwise distances between the two sets |
| `pdist(coords)` | One `(N, 3)` array | All within-set pairwise distances |
| `find_clashes(mol, sel1, sel2)` | Molecule + two selections | Returns clash pairs, distances, and overlap amounts |

## Common variations

```python
# Within-selection pairwise distances
from moleculekit.distance import pdist

ca_idx = mol.atomselect("name CA", indexes=True)
within = pdist(mol.coords[ca_idx, :, 0])
```

```python
# Detect steric clashes between protein and ligand
from moleculekit.distance import find_clashes

clashes, distances, overlaps = find_clashes(mol, sel1="protein", sel2="resname BEN")
print(f"{len(clashes)} clash pairs found")
```

## Gotchas

- {py:func}`~moleculekit.distance.cdist` and {py:func}`~moleculekit.distance.pdist` operate on raw coordinate slices, not {py:class}`~moleculekit.molecule.Molecule` objects — extract the coordinate array first with `mol.coords[idx, :, frame]`.
- Distance arrays are per-frame; loop over frames explicitly for multi-frame analysis.
- Large selections produce large distance matrices — slice selections or frames before calling to avoid memory issues.
- `find_clashes` includes a distance-based bond guesser by default; pass `guess_bonds=False` when `mol.bonds` is already complete to avoid overhead.

## See also

- [How to select atoms](select-atoms.md)
- [How to compute RMSD and RMSF](compute-rmsd-rmsf.md)
- [How to compute projections](compute-projections.md)
