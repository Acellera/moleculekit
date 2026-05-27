# How to compute protein–ligand interactions

## Goal

Detect hydrogen bonds, π–π stacking, cation–π, and σ-hole interactions between a protein receptor and a small-molecule ligand.

## Minimal example

```python
import numpy as np
from moleculekit.molecule import Molecule
from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.interactions.interactions import (
    hbonds_calculate,
    get_ligand_donors_acceptors,
)

mol = Molecule("3PTB")
mol.templateResidueFromSmiles(
    mol.resname == "BEN",
    smiles="[NH2+]=C(N)c1ccccc1",
    addHs=True,
)

# Build a SmallMol view of the ligand, then ask the library helper for
# donor/acceptor indices in the parent Molecule's atom numbering.
lig = SmallMol(mol.copy(sel="resname BEN"))
start_idx = int(mol.atomselect("resname BEN", indexes=True)[0])
donors, acceptors = get_ligand_donors_acceptors(lig, start_idx=start_idx)

# For non-periodic structures hbonds_calculate needs a zero box.
mol.box = np.zeros((3, mol.numFrames), dtype=np.float32)
hbonds = hbonds_calculate(mol, donors, acceptors, sel2="protein")
print(f"H-bonds in frame 0: {len(hbonds[0])}")
```

## Parameters that matter

| Function/parameter | What it does |
|---|---|
| {py:class}`~moleculekit.smallmol.smallmol.SmallMol` | Lightweight RDKit-backed view of the ligand used by the helper to walk the bond graph. |
| `start_idx` | Offset of the first ligand atom in the parent `Molecule`; the helper adds it so the returned indices reference the parent. |
| `get_ligand_donors_acceptors(smol, start_idx=...)` | Returns `(donors, acceptors)`. `donors` is a `(N, 2)` array of `[heavy_atom_idx, hydrogen_idx]` pairs; `acceptors` is a 1D array of heavy-atom indices. |
| `hbonds_calculate(mol, donors, acceptors, sel2=...)` | Returns a list (one entry per frame) of H-bonds; each entry is a list of `(donor_heavy, donor_h, acceptor)` triples. |

## Common variations

```python
# Convert the per-frame list of triples into a single numpy array of donors,
# Hs, and acceptors for frame 0.
frame0 = np.array(hbonds[0])   # shape (n_hbonds, 3)
print("donor heavy atoms, Hs, acceptors:\n", frame0)
```

```python
# Full interaction set (H-bond + π–π + cation–π + σ-hole) via the high-level API
from moleculekit.interactions.interactions import (
    pipi_calculate,
    cationpi_calculate,
    sigmahole_calculate,
    get_protein_rings,
)

rings = get_protein_rings(mol)
```

## Gotchas

- Reasonable hydrogens must be present — run {py:func}`~moleculekit.tools.preparation.systemPrepare` (or pass `addHs=True` to {py:meth}`~moleculekit.molecule.Molecule.templateResidueFromSmiles`) before this step.
- {py:func}`~moleculekit.interactions.interactions.get_ligand_donors_acceptors` returns donor *pairs* `[heavy, H]`, **not** lone heavy atoms. Building your own pairs as `(heavy, heavy)` silently produces wrong H-bond results because `hbonds_calculate` reads column 1 as the hydrogen index.
- `start_idx` is critical when the ligand atoms do not start at index 0 in the parent molecule — the helper offsets every returned index by `start_idx`.
- For non-periodic structures set `mol.box` to a zero array before calling `hbonds_calculate`.

## See also

- [How to convert a Molecule to RDKit or OpenFF](convert-to-rdkit-and-openff.md)
- [How to compute distances and contacts](compute-distances-and-contacts.md)
