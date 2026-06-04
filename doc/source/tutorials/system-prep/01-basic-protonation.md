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

# Basic protonation

**You will learn:** how to add hydrogens to a protein at a chosen pH, inspect the resulting protonation-state table, and write the prepared system to disk.

**Prerequisites:**
- The [First molecule](../01-first-molecule.md) tutorial.
- PDB2PQR and PROPKA installed (they ship as moleculekit dependencies).

## Setup

```{code-cell} python
import pandas as pd
from moleculekit.molecule import Molecule
from moleculekit.tools.preparation import systemPrepare

mol = Molecule("3PTB")
```

```{code-cell} python
:tags: [remove-input]
from acellera_docs_theme.molstar import show3d
```

3PTB is bovine trypsin with a benzamidine ligand in the active site.

```{code-cell} python
:tags: [remove-input]
show3d(mol)
```

## Step 1 — Run systemPrepare at pH 7.4

```{code-cell} python
pmol, specs, details = systemPrepare(mol, pH=7.4, return_details=True)
```

```{code-cell} python
:tags: [remove-input]
show3d(pmol, representations=[{"sel": "protein", "type": "ball_and_stick", "size_factor": 0.6}])
```

The call returns a 3-tuple: `pmol`, `specs`, `details`. `pmol` is a **new** {py:class}`~moleculekit.molecule.Molecule` — the input `mol` is not mutated. `specs` is the list of detected non-standard-residue specs that the call applied (same type as returned by {py:func}`~moleculekit.tools.nonstandard_residues.detectNonStandardResidues`); pass it back to a later `systemPrepare` call if you need to repeat the run, or inspect it to audit which residues were renamed. `details` is a `pandas.DataFrame` with one row per titratable residue; columns include `resname`, `resid`, `chain`, `segid`, `pKa`, `protonation`, and `buried`. The function adds hydrogens, runs PROPKA to predict pKa values, and titrates each titratable residue accordingly.

## Step 2 — Inspect protonation states

```{code-cell} python
details[["resname", "resid", "protonation", "pKa"]].head(10)
```

Each row shows the assigned protonation form for one residue. Histidines appear as `HID`, `HIE`, or `HIP` depending on tautomer or charge; aspartates as `ASP` (deprotonated) or `ASH` (protonated); glutamates as `GLU` or `GLH`; cysteines as `CYS` or `CYM`; lysines as `LYS` or `LYN`.

Residues whose predicted pKa falls within 2 units of the target pH are the most sensitive to the pH choice — flipping them would change their protonation state if pH moved a unit or two:

```{code-cell} python
import numpy as np

pkas = pd.to_numeric(details["pKa"], errors="coerce")
near_pka = details[np.abs(pkas - 7.4) < 2.0]
near_pka[["resname", "resid", "chain", "protonation", "pKa"]]
```

```{code-cell} python
:tags: [remove-input]
show3d(pmol, sel="not water", representations=[{"sel": "chain A and resid 39 57 70", "type": "ball_and_stick", "size_factor": 0.6}], focus="chain A and resid 39 57 70")
```

These residues would flip protonation state if pH moved a unit or two from 7.4.

## Step 3 — Skip titration entirely

```{code-cell} python
pmol_no_titr, _ = systemPrepare(mol, titration=False)
```

`titration=False` skips PROPKA. Hydrogens are still added by PDB2PQR, but every titratable residue gets the standard protonation form at default pH with no per-residue prediction. Use this when you already know the protonation states you want, or when you will set them yourself via `force_protonation`.

## Step 4 — Write the prepared structure

```{code-cell} python
pmol.write("trypsin_prepared.cif")
```

mmCIF is the recommended output format here: it preserves the bonds, bond orders, and formal charges that `systemPrepare` just established. Reload with {py:class}`~moleculekit.molecule.Molecule`("trypsin_prepared.cif") to round-trip.

## Recap

- {py:func}`~moleculekit.tools.preparation.systemPrepare` adds hydrogens and assigns protonation states to `mol` at the chosen `pH` in one call.
- `return_details=True` gives you a per-titratable-residue table (a `pandas.DataFrame`) for inspection.
- `titration=False` skips PROPKA when you do not need pKa prediction.

## Next

- [Non-standard residues](02-non-standard-residues.md)
- [System-preparation pipeline](../../explanation/system-preparation-pipeline.md)
