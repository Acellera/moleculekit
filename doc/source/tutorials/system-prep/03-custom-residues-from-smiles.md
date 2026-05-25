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

## Step 1 — Load a structure with several non-canonical residues

We use **5VBL**, a zinc metalloprotease structure that carries a bound peptide inhibitor built from **five different non-canonical amino acids** (HRG, ALC, OIC, NLE, and 200) plus a free OLC monoglyceride ligand. The peptide is cyclized through an isopeptide bond, which makes this a good stress test for templating + bond preservation. A Zn²⁺ ion is also present but does not need templating.

After loading, strip any hydrogens so the template's `addHs=True` becomes the single, deterministic source of hydrogens:

```{code-cell} python
mol = Molecule("5VBL")
mol.remove("hydrogen")
```

If the input is already H-free the return is just an empty `array([], dtype=uint32)`; the call is a defensive no-op in that case.

## Step 2 — Inspect what needs templating

```{code-cell} python
sorted(set(mol.resname))
```

You will see the five non-canonical resnames `HRG`, `ALC`, `OIC`, `NLE`, `200` alongside the standard amino-acid codes, plus the `OLC` free ligand. Each non-canonical residue needs a SMILES template so the bond orders, formal charges, and hydrogens come out right.

## Step 3 — Template every NCAA from its RCSB SMILES

The RCSB-style SMILES for a canonical amino acid is written as the free form (with N and C terminal groups). When the residue sits inside a peptide chain, `templateResidueFromSmiles` automatically strips the unmatched terminal heavy atoms (the OXT and one peptide-NH) before the MCS match — no manual trimming is required.

```{code-cell} python
smiles = {
    "HRG": "C(CCNC(=N)N)C[C@@H](C(=O)O)N",         # homoarginine
    "ALC": "C1CCC(CC1)C[C@@H](C(=O)O)N",            # cyclohexyl-Ala
    "OIC": "C1CC[C@H]2[C@@H](C1)C[C@H](N2)C(=O)O",  # octahydroindole-2-carboxylic acid
    "NLE": "CCCC[C@@H](C(=O)O)N",                   # norleucine
    "200": "c1cc(ccc1C[C@@H](C(=O)O)N)Cl",          # 4-chloro-Phe
    "OLC": "CCCCCCCC(=O)OC[C@H](O)CO",              # monoglyceride free ligand
}

for resname, smi in smiles.items():
    mask = mol.resname == resname
    if mask.any():
        mol.templateResidueFromSmiles(mask, smiles=smi, addHs=True)
```

Each call mutates `mol` in place. Cross-residue covalent bonds — the peptide bonds connecting the NCAAs to their neighbours and the isopeptide cyclization bond — are detected automatically from `mol.bonds`, and the boundary atoms' H counts are reduced so they are not over-protonated. OLC is a free ligand with no covalent bonds to the protein, so its template is applied without any boundary adjustments. When a residue appears multiple times in the structure, every copy is templated individually with the same SMILES.

```{code-cell} python
import numpy as np
ncaa_sel = "resname " + " ".join(f"'{r}'" for r in smiles)
np.unique(mol.resname[mol.atomselect(ncaa_sel)], return_counts=True)
```

The per-resname atom counts after templating — each NCAA now carries heavy atoms + the hydrogens the SMILES specified.

## Step 4 — Run systemPrepare

```{code-cell} python
specs = detectNonStandardResidues(mol)
pmol, applied_specs = systemPrepare(mol, detect_specs=specs, verbose=False)
```

`systemPrepare` captures all non-peptidic bonds (including the isopeptide cyclization) before invoking PDB2PQR, then restores them on the prepared structure. Without templating, those bonds would either be silently dropped or trigger valence errors during hydrogenation.

```{code-cell} python
pmol.atomselect(ncaa_sel).sum()
```

The five NCAAs and the OLC free ligand all survive the preparation pipeline with their full heavy-atom topology and hydrogens.

## Recap

- {py:meth}`~moleculekit.molecule.Molecule.templateResidueFromSmiles` transfers bond orders, formal charges, and hydrogens from a SMILES template onto a residue's atoms by MCS matching.
- Strip Hs first, then template with `addHs=True` to get a single, deterministic hydrogen pattern.
- One SMILES per residue type; the templater handles every copy automatically and trims terminal atoms (OXT, terminal NH) for mid-chain residues.
- Cross-residue covalent bonds — peptide bonds, glycosidic bonds, isopeptide bonds — are detected automatically; the boundary atom's H count is corrected so it is not over-protonated.

## Next

- [Mutation, gap closing, segmentation](04-mutation-gap-closing-segmentation.md)
- [System-preparation pipeline](../../explanation/system-preparation-pipeline.md)
