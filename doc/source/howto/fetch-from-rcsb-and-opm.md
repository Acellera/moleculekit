# How to fetch from RCSB and OPM

## Goal

Programmatically download structures from the RCSB PDB and membrane-oriented coordinates from the Orientations of Proteins in Membranes (OPM) database.

## Minimal example

```python
from moleculekit.molecule import Molecule

# Download directly from RCSB by 4-character PDB ID
mol = Molecule("3PTB")
print(mol.numAtoms)
```

## Parameters that matter

| Function | Key parameters | What it does |
|---|---|---|
| `Molecule(pdbid)` | 4-character string | Fetches and parses the PDB entry |
| `rcsbFindLigands(pdbid)` | `pdbid` | Returns a list of ligand component IDs for that entry |
| `get_opm_pdb(pdbid, keep=False, keepaltloc="A", validateElements=False)` | `pdbid`, `keep` | Downloads the OPM-oriented structure; `keep=True` also returns dummy membrane atoms |
| `align_to_opm(mol, molsel="all", maxalignments=3, opmid=None, macrotype="protein")` | `mol`, `opmid` | Aligns `mol` to its OPM counterpart by sequence search |

## Common variations

```python
# List the ligands bound in a structure, then fetch
from moleculekit.rcsb import rcsbFindLigands

ligands = rcsbFindLigands("3PTB")
print(ligands)

mol = Molecule("3PTB")
```

```python
# Fetch a membrane protein in its OPM orientation
from moleculekit.opm import get_opm_pdb

mol, thickness = get_opm_pdb("1BL8")
```

```python
# Align your own structure to its OPM equivalent
from moleculekit.opm import align_to_opm

mol = Molecule("my_structure.pdb")
mol_opm = align_to_opm(mol, maxalignments=3)
```

## Gotchas

- RCSB downloads respect the server rate limits; avoid hammering the API in tight loops.
- Set the `LOCAL_PDB_REPO` environment variable to a local PDB mirror directory to avoid repeated network downloads.
- OPM membership requires a known PDB ID or a successful BLAST sequence alignment; structures absent from OPM will raise or return `None`.
- {py:func}`~moleculekit.opm.get_opm_pdb` with `keep=False` (default) strips the dummy membrane atoms that OPM adds; pass `keep=True` if you need them for visualization.

## See also

- [How to read a structure](read-a-structure.md)
- [How to align structures](align-structures.md)
