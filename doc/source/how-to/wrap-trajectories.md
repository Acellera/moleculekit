# How to wrap trajectories

## Goal

Re-image atoms into the central periodic box so that molecules appear whole rather than split across box boundaries.

## Minimal example

```python
from moleculekit.molecule import Molecule

mol = Molecule("topology.psf")
mol.read("trajectory.xtc")

# Wrap and center the box on the protein
mol.wrap("protein")
```

## Parameters that matter

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `wrapsel` | `str` | `"all"` | Atom selection that defines the center of the wrapping box |
| `fileBonds` | `bool` | `True` | Use bonds from the loaded topology file |
| `guessBonds` | `bool` | `False` | Also include distance-guessed bonds in the bond graph used for wrap grouping |
| `wrapcenter` | `array_like` | `None` | Explicit center to wrap around. Usually leave as `None` and use `wrapsel` instead, so the center tracks the selection across frames |
| `unitcell` | `str` | `"rectangular"` | Unit-cell representation for triclinic boxes: `"rectangular"`, `"triclinic"`, or `"compact"`. No effect on rectangular boxes |

## Common variations

```python
# Wrap around a ligand instead of the whole protein
mol.wrap("resname LIG")
```

```python
# Topology file is missing some bonds (e.g. small molecules in a plain PDB) —
# merge in distance-guessed bonds so the wrap groups stay intact
mol.wrap("protein", guessBonds=True)
```

## Gotchas

- Wrapping requires a valid `mol.box` array (one column per frame). If `mol.box` is all zeros, `wrap` logs a warning and returns without modifying coordinates — the original `mol` is unchanged.
- Non-orthorhombic (triclinic) boxes use `mol.boxangles` in addition to `mol.box`; make sure both are read from the trajectory. For triclinic boxes you can also choose how the cell is wrapped with the `unitcell` parameter — `"rectangular"` (default, a parallelepiped), `"triclinic"`, or `"compact"` (minimum-volume shape, handy for visualizing truncated octahedra or rhombic dodecahedra).
- `wrap` operates on all frames in place.
- Molecules that are already centered in the box will be re-wrapped (the operation is applied regardless).

## See also

- [How to read a trajectory](read-a-trajectory.md)
- [How to align structures](align-structures.md)
