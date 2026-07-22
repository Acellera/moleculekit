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
| `rcsbFetchLigandSmiles(comp_id, stereo=True, program="OpenEye")` | `comp_id`, `stereo`, `program` | Returns a SMILES string for a chemical-component (CCD) code; `stereo=False` drops stereochemistry, `program` picks the toolkit (`OpenEye`, `CACTVS`, `ACDLabs`) whose descriptor to use |
| `fetchResidueCIF(resname, outdir, overwrite=False)` | `resname`, `outdir`, `overwrite` | Downloads a component's reference structure from the Chemical Component Dictionary and writes `<resname>.cif` into `outdir`; skips the download if the file already exists unless `overwrite=True` |
| `rcsbFindMutatedResidues(pdbid)` | `pdbid` | Returns a dict mapping each modified/mutated residue name in the entry to its parent standard residue (e.g. `{"MSE": "MET"}`) |
| `resolveFullSequences(mol, pdbid=None)` | `mol`, `pdbid` | Returns `{chain: {"sequence": str, "source": str, "identity": float}}`, the full deposited sequence per protein chain; uses the exact entity sequence when `pdbid` is given, otherwise falls back to `rcsbSequenceSearch` |
| `rcsbSequenceSearch(sequence, identity_cutoff=0.9, rows=10)` | `sequence`, `identity_cutoff`, `rows` | Searches RCSB's sequence-similarity service and returns a list of `{"polymer_entity_id", "identity", "score"}` hits, best-first |
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
results = align_to_opm(mol, maxalignments=3)

# results is a list, one entry per OPM hit. Each entry has the OPM PDB ID,
# the membrane thickness, and a list of high-scoring sequence pairs (HSPs)
# whose aligned_mol is `mol` re-imaged into the OPM frame.
for hit in results:
    print(hit["pdbid"], "thickness:", hit["thickness"])
    for hsp in hit["hsps"]:
        aligned_mol = hsp["aligned_mol"]   # Molecule, oriented in the OPM membrane frame
        print(f"  TM={hsp['TM-Score']:.2f}  RMSD={hsp['Common RMSD']:.2f} Å")
```

```python
# Fetch a component's reference SMILES / CIF to build a residue template
from moleculekit.rcsb import rcsbFetchLigandSmiles, fetchResidueCIF

smiles = rcsbFetchLigandSmiles("HRG")           # RCSB canonical SMILES for a CCD code
cif_path = fetchResidueCIF("HRG", outdir=".")   # component CIF (bonds, charges) on disk
```

```python
# Resolve full deposited sequences (input to detectSequenceGaps; see tutorial 05)
from moleculekit.molecule import Molecule
from moleculekit.rcsb import resolveFullSequences

mol = Molecule("5VQ2")
resolved = resolveFullSequences(mol, "5VQ2")
sequences = {ch: v["sequence"] for ch, v in resolved.items()}
```

## Gotchas

- RCSB downloads respect the server rate limits; avoid hammering the API in tight loops.
- Set the `LOCAL_PDB_REPO` environment variable to a local PDB mirror directory to avoid repeated network downloads.
- OPM membership requires a known PDB ID or a successful BLAST sequence alignment; when nothing matches, {py:func}`~moleculekit.opm.align_to_opm` returns an **empty list** (not `None`).
- {py:func}`~moleculekit.opm.align_to_opm` returns a `list[dict]` — the aligned `Molecule` objects live under `hit["hsps"][j]["aligned_mol"]`, not at the top level.
- {py:func}`~moleculekit.opm.get_opm_pdb` with `keep=False` (default) strips the dummy membrane atoms that OPM adds; pass `keep=True` if you need them for visualization.

## See also

- [How to read a structure](read-a-structure.md)
- [How to align structures](align-structures.md)
- [Filling missing loops](../tutorials/system-prep/05-filling-missing-loops.md)
