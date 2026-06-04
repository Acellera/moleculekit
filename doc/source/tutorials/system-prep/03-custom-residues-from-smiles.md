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

# Custom residues from SMILES

**You will learn:** how to teach {py:class}`~moleculekit.molecule.Molecule` about a non-canonical residue's bond orders, formal charges, and hydrogens by templating it from a SMILES string, then run {py:func}`~moleculekit.tools.preparation.systemPrepare` over the result.

**Prerequisites:**
- The [Non-standard residues](02-non-standard-residues.md) tutorial.

## Setup

```{code-cell} python
from moleculekit.molecule import Molecule
from moleculekit.tools.preparation import systemPrepare
from moleculekit.tools.nonstandard_residues import detectNonStandardResidues
```

```{code-cell} python
:tags: [remove-input]
from acellera_docs_theme.molstar import show3d
```

## Step 1 — Load a structure with several non-canonical residues

We use **5VBL**, a zinc metalloprotease structure that carries a bound peptide inhibitor built from **five different non-canonical amino acids** (HRG, ALC, OIC, NLE, and 200), plus a free **OLC** monoacylglycerol lipid (monoolein) from lipidic-cubic-phase crystallization. The peptide is cyclized through an isopeptide bond, which makes this a good stress test for templating + bond preservation. A Zn²⁺ ion is also present but does not need templating.

Load the structure:

```{code-cell} python
mol = Molecule("5VBL")
```

```{code-cell} python
:tags: [remove-input]
show3d(mol)
```

## Step 2 — Find what needs templating

```{code-cell} python
specs = detectNonStandardResidues(mol)
sorted(set(s.resname for s in specs))
```

Rather than eyeballing every residue name, {py:func}`~moleculekit.tools.nonstandard_residues.detectNonStandardResidues` walks the bond graph and returns a spec for each residue that needs special handling. Here it flags the five non-canonical amino acids (`HRG`, `ALC`, `OIC`, `NLE`, `200`), a free `OLC` residue, and the two canonical residues (`GLU`, `LYS`) that form the isopeptide cyclization bond.

`OLC` is monoolein — a monoacylglycerol lipid from lipidic-cubic-phase crystallization, not part of the system we want to model. Drop it from both the structure and the spec list before going further:

```{code-cell} python
mol.remove("resname OLC")
specs = [s for s in specs if s.resname != "OLC"]
```

The five non-canonical amino acids each need a SMILES template so their bond orders, formal charges, and hydrogens come out right (next step); the canonical isopeptide partners (`GLU`, `LYS`) don't need a template — {py:func}`~moleculekit.tools.preparation.systemPrepare` simply preserves their crosslink.

## Step 3 — Template every NCAA from its RCSB SMILES

The RCSB-style SMILES for a canonical amino acid is written as the free form (with N and C terminal groups). When the residue sits inside a peptide chain, `templateResidueFromSmiles` automatically strips the unmatched terminal heavy atoms (the OXT and one peptide-NH) before the MCS match — no manual trimming is required.

```{code-cell} python
smiles = {
    "HRG": "C(CCNC(=N)N)C[C@@H](C(=O)O)N",         # homoarginine
    "ALC": "C1CCC(CC1)C[C@@H](C(=O)O)N",            # cyclohexyl-Ala
    "OIC": "C1CC[C@H]2[C@@H](C1)C[C@H](N2)C(=O)O",  # octahydroindole-2-carboxylic acid
    "NLE": "CCCC[C@@H](C(=O)O)N",                   # norleucine
    "200": "c1cc(ccc1C[C@@H](C(=O)O)N)Cl",          # 4-chloro-Phe
}

for resname, smi in smiles.items():
    mask = mol.resname == resname
    if mask.any():
        mol.templateResidueFromSmiles(mask, smiles=smi, addHs=True)
```

Each call mutates `mol` in place. Cross-residue covalent bonds — the peptide bonds connecting the NCAAs to their neighbours and the isopeptide cyclization bond — are detected automatically from `mol.bonds`, and the boundary atoms' H counts are reduced so they are not over-protonated. When a residue appears multiple times in the structure, every copy is templated individually with the same SMILES.

```{code-cell} python
import numpy as np
{r: int(np.sum(mol.resname == r)) for r in smiles}
```

```{code-cell} python
:tags: [remove-input]
show3d(mol, representations=[{"sel": "resname HRG ALC OIC NLE '200'", "type": "ball_and_stick"}], focus="resname HRG ALC OIC NLE '200'")
```

The per-resname atom counts after templating — each NCAA now carries heavy atoms + the hydrogens the SMILES specified.

## Step 4 — Run systemPrepare

```{code-cell} python
pmol, applied_specs = systemPrepare(mol, detect_specs=specs, verbose=False)
```

`systemPrepare` captures all non-peptidic bonds (including the isopeptide cyclization) before invoking PDB2PQR, then restores them on the prepared structure. Without templating, those bonds would either be silently dropped or trigger valence errors during hydrogenation.

```{code-cell} python
{r: int(np.sum(pmol.resname == r)) for r in smiles}
```

The five NCAAs all survive the preparation pipeline with their full heavy-atom topology and hydrogens.

## Recap

- {py:meth}`~moleculekit.molecule.Molecule.templateResidueFromSmiles` transfers bond orders, formal charges, and hydrogens from a SMILES template onto a residue's atoms by MCS matching.
- It removes any hydrogens already on the matched residue and re-adds them from the SMILES (`addHs=True`), so the hydrogen pattern is deterministic — no manual pre-stripping needed.
- One SMILES per residue type; the templater handles every copy automatically and trims terminal atoms (OXT, terminal NH) for mid-chain residues.
- Cross-residue covalent bonds — peptide bonds, glycosidic bonds, isopeptide bonds — are detected automatically; the boundary atom's H count is corrected so it is not over-protonated.

## Next

- [Mutation, gap closing, segmentation](04-mutation-gap-closing-segmentation.md)
- [System-preparation pipeline](../../explanation/system-preparation-pipeline.md)
