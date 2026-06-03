# How to set and get properties

## Goal

Read or write per-atom attributes (such as residue name, chain, or charge) for a selection of atoms.

## Minimal example

```python
from moleculekit.molecule import Molecule

mol = Molecule("3PTB")

# Read the residue names of all Cα atoms
names = mol.get("resname", sel="protein and name CA")

# Set the chain of all atoms with resid 100 to "B"
mol.set("chain", "B", sel=mol.resid == 100)
```

## Parameters that matter

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `field` | `str` | required | The atom attribute to read or write |
| `sel` | `str`, boolean `np.ndarray`, or `None` | `None` (all) | Which atoms to read/write |
| `fileBonds` | `bool` | `True` | Use file bonds for bond-dependent selectors (`get` only) |
| `guessBonds` | `bool` | `True` | Fall back to guessed bonds for bond-dependent selectors (`get` only) |

## Common variations

```python
# Read a field for the entire molecule
all_resnames = mol.get("resname")
```

```python
# Read with a boolean mask — direct array indexing is equivalent to .get
ala_mask = mol.resname == "ALA"
ala_names = mol.name[ala_mask]          # equivalent to mol.get("name", sel=ala_mask)
ala_coords = mol.coords[ala_mask]       # full (n_ALA, 3, numFrames) — note .get returns one frame
```

```python
# Write with a boolean mask — direct assignment is equivalent to .set
mol.charge[ala_mask] = 0.0              # equivalent to mol.set("charge", 0.0, sel=ala_mask)
```

When you already hold a mask, the direct-indexing form is shorter and skips the `field` lookup string. Reach for `get`/`set` when the natural form of your selection is a string (atomselect syntax) you want re-parsed at call time.

## Gotchas

- Common public fields: `name`, `resname`, `resid`, `chain`, `segid`, `element`, `charge`, `masses`, `coords`.
- For `coords`, `get` returns the coordinates of the current active frame (`mol.frame`), not all frames. Direct indexing (`mol.coords[mask]`) returns the full `(n_sel, 3, numFrames)` slice.
- Setting `coords` via `set` only updates the current frame; direct indexing again gives you the whole trajectory.

## See also

- [How to select atoms](select-atoms.md)
- [How to filter and remove atoms](filter-and-remove.md)
