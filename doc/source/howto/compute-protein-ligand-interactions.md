# How to compute protein–ligand interactions

## Goal

Detect hydrogen bonds, π–π stacking, cation–π, and σ-hole interactions between a protein receptor and a small-molecule ligand.

## Minimal example

```python
import numpy as np
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os
from moleculekit.molecule import Molecule
from moleculekit.interactions.interactions import hbonds_calculate

mol = Molecule("3PTB")
mol.templateResidueFromSmiles(
    mol.resname == "BEN",
    smiles="NC(=N)c1ccccc1",
    addHs=True,
)

# Convert the ligand to RDKit, then ask RDKit which atoms are H-bond
# donors and acceptors.
lig = mol.copy(sel="resname BEN")
rdlig = lig.toRDKitMol(sanitize=True)

fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
factory = ChemicalFeatures.BuildFeatureFactory(fdef)
features = factory.GetFeaturesForMol(rdlig)

start_idx = int(mol.atomselect("resname BEN", indexes=True)[0])
acceptors = [idx + start_idx for f in features if f.GetFamily() == "Acceptor"
             for idx in f.GetAtomIds()]
donor_pairs = [(idx + start_idx, idx + start_idx)
               for f in features if f.GetFamily() == "Donor"
               for idx in f.GetAtomIds()]

# For non-periodic structures hbonds_calculate needs a zero box.
mol.box = np.zeros((3, mol.numFrames), dtype=np.float32)
hbonds = hbonds_calculate(mol, np.array(donor_pairs), np.array(acceptors), sel2="protein")
print(f"H-bonds in frame 0: {len(hbonds[0])}")
```

## Parameters that matter

| Function/parameter | What it does |
|---|---|
| `toRDKitMol(sanitize=True)` | Convert the ligand selection to an RDKit Mol for feature detection. |
| `RDConfig.RDDataDir / "BaseFeatures.fdef"` | RDKit's standard donor/acceptor feature definition file. |
| `start_idx` | Offset of the first ligand atom in the parent `Molecule`; required when the ligand does not start at atom 0. |
| `hbonds_calculate(mol, donors, acceptors, sel2=...)` | Returns a list (one entry per frame) of H-bond arrays. |

## Common variations

```python
# Visualise H-bonds in VMD (requires VMD on PATH)
from moleculekit.interactions.interactions import view_hbonds

view_hbonds(mol, hbonds)
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

- Reasonable hydrogens must be present — run {py:func}`~moleculekit.tools.preparation.systemPrepare` before this step if the input lacks explicit H atoms.
- Feature detection uses RDKit through {py:meth}`~moleculekit.molecule.Molecule.toRDKitMol`; make sure RDKit is installed.
- `start_idx` is critical when the ligand atoms do not start at index 0 in the parent molecule.
- For non-periodic structures set `mol.box` to a zero array before calling `hbonds_calculate`.

## See also

- [How to convert a Molecule to RDKit or OpenFF](convert-to-rdkit-and-openff.md)
- [How to compute distances and contacts](compute-distances-and-contacts.md)
