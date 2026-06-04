# How to append and merge molecules

## Goal

Combine two {py:class}`~moleculekit.molecule.Molecule` instances into one, either by appending at the end or inserting at a specific position.

## Minimal example

```python
from moleculekit.molecule import Molecule

receptor = Molecule("3PTB")
receptor.filter("protein")

ligand = Molecule("3PTB")
ligand.filter("resname BEN")

# Append ligand after the receptor atoms
receptor.append(ligand)
print(receptor.numAtoms)
```

## Parameters that matter

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `mol` | {py:class}`~moleculekit.molecule.Molecule` | required | The molecule to append or insert |
| `collisions` | `bool` | `False` | Remove residues from `mol` that clash with existing atoms |
| `coldist` | `float` | `1.3` | Distance threshold in Å for collision detection |

## Common variations

```python
# Insert a molecule at the beginning (index 0)
water = Molecule("water.pdb")
receptor.insert(water, index=0)
```

```python
# Append with clash removal to avoid overlapping atoms
receptor.append(ligand, collisions=True, coldist=1.5)
```

## Gotchas

- Appending molecules with a different number of frames raises an error — make sure frame counts match or reduce to a single frame first.
- Segment IDs and chain IDs from both molecules are preserved as-is; collisions in these labels are not auto-resolved.
- {py:meth}`~moleculekit.molecule.Molecule.append` is equivalent to {py:meth}`~moleculekit.molecule.Molecule.insert` with `mol` and `index=self.numAtoms`.
- After appending, existing boolean masks are stale because the atom count has changed (see [Mask and index substitution](../explanation/atom-selection-language.md#mask-and-index-substitution)).

## See also

- [How to filter and remove atoms](filter-and-remove.md)
- [How to assign segments and chains](assign-segments-and-chains.md)
