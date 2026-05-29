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

# Mutation, gap closing, and auto-segmentation

**You will learn:** how to fix missing chain/segment IDs with {py:func}`~moleculekit.tools.autosegment.autoSegment`, mutate a residue using the Dunbrack rotamer library, and (with the ProMod3 Singularity image) close missing-residue gaps — the surrounding workflow you typically combine with {py:func}`~moleculekit.tools.preparation.systemPrepare`.

**Prerequisites:**
- The [Custom residues from SMILES](03-custom-residues-from-smiles.md) tutorial.

## Setup

```{code-cell} python
from moleculekit.molecule import Molecule
from moleculekit.tools.preparation import systemPrepare
from moleculekit.tools.autosegment import autoSegment
from moleculekit.tools.modelling import model_gaps
```

## Step 1 — Auto-segment a structure with a chain break

We use **1ITG**, an integrin headpiece that has 13 residues missing from the deposited structure (residues 141–153 are absent from the PDB file). After loading and filtering to protein only, the molecule carries a single chain letter but lacks meaningful segids:

```{code-cell} python
mol = Molecule("1itg")
mol.filter("protein")
mol.segid[:] = ""
mol.chain[:] = ""
```

Loading drops 67 non-protein atoms; clearing segids and chain IDs gives us a clean slate that mirrors what you receive from many homology-model or trajectory outputs.

```{code-cell} python
mol = autoSegment(mol, sel="protein", fields=("chain", "segid"))
sorted(set(zip(mol.chain, mol.segid)))
```

{py:func}`~moleculekit.tools.autosegment.autoSegment` detects that the backbone is broken between GLY 140 and MET 154 (the flanking residues of the gap) — their `C–N` distance far exceeds the peptide-bond cutoff (`protein_cutoff`, 2 Å by default) — and so it creates two independent segments: `P0` on chain A (residues 55–140) and `P1` on chain B (residues 154–209). Both the `chain` and `segid` fields are now consistent, which avoids warnings during {py:func}`~moleculekit.tools.preparation.systemPrepare`.

## Step 2 — Mutate a residue with the "best" rotamer

We mutate **GLN 95** (a surface-exposed glutamine in segment P0) to alanine. The `"best"` rotamer mode queries the Dunbrack backbone-dependent rotamer library for the phi/psi bin of that residue and picks the rotamer with the lowest van der Waals clash energy against the surrounding atoms:

```{code-cell} python
mut = mol.copy()

print("before:", set(mut.resname[(mut.chain == "A") & (mut.resid == 95)]))

mut.mutateResidue("chain A and resid 95", "ALA", rotamer_mode="best")

print("after :", set(mut.resname[(mut.chain == "A") & (mut.resid == 95)]))
```

`before: {'GLN'}` then `after: {'ALA'}` — the mutation flipped the residue identity. {py:meth}`~moleculekit.molecule.Molecule.mutateResidue` modifies the molecule in place: the old sidechain is stripped, an ideal ALA template is Kabsch-aligned onto the backbone (N, CA, C), and the new sidechain is placed. Notice that the two masks are recomputed against `mut` *after* the mutation, not reused from before — the mutation shrinks the atom array (GLN has more sidechain atoms than ALA), and a precomputed mask would now be the wrong length (see the warning in the [Atom selection tutorial](../02-atom-selection.md#for-developers-bypass-the-parser-with-masks)).

## Step 3 — Compare "random" vs "best", and add minimization

`rotamer_mode="random"` samples a rotamer weighted by its library probability rather than scoring clash energy, which is faster but may leave residual steric strain:

```{code-cell} python
mut_random = mol.copy()
mut_random.mutateResidue("chain A and resid 95", "ALA", rotamer_mode="random")
```

Use `"random"` when throughput matters more than placement quality.

Adding `minimize=True` invokes an OpenMM soft-potential minimization after rotamer placement that relaxes sidechain dihedrals to remove residual strain. When OpenMM is not installed the step is silently skipped:

```{code-cell} python
mut_min = mol.copy()
mut_min.mutateResidue("chain A and resid 95", "ALA", rotamer_mode="best", minimize=True)
```

`minimize=True` is the safest choice for downstream MD setup; without it the rotamer placement is still physically reasonable but may retain small clashes.

## Step 4 — Close the gap with ProMod3

{py:func}`~moleculekit.tools.modelling.model_gaps` calls ProMod3 inside a Singularity/Apptainer container to reconstruct the missing loop. This cell is **skipped in CI** because it requires the ProMod3 image:

```{code-cell} python
:tags: [skip-execution]
modeled_p0 = model_gaps(
    mol,
    sequence="DCSPGIWQLDCTHLEGKVILVAVHVASGYIEAEVIPAETGQETAYFLLKLAGRWPVKTVHTDNGSNFTSTTVKAACWWAGIKQEFGIPNKELKKIIGQVRDQAEHLKTAVQMAVFIHNKKRKGGIGGYSAGERIVDIIATDIQ",
    segid="P0",
    promod_img="/path/to/promod3.sif",
)
```

{py:func}`~moleculekit.tools.modelling.model_gaps` aligns `sequence` against the atoms present in segment `P0`, writes a temporary FASTA alignment, runs ProMod3's loop-modelling pipeline inside the container, and returns a new {py:class}`~moleculekit.molecule.Molecule` with the gap filled.

ProMod3 is available as an Apptainer/Singularity image from the OpenStructure project — follow the instructions at <https://openstructure.org/promod3/> to obtain `promod3.sif`. If neither `apptainer` nor `singularity` is on the `PATH`, {py:func}`~moleculekit.tools.modelling.model_gaps` raises a `RuntimeError` immediately; there is no fallback gap-filling path.

## Step 5 — Wrap up with systemPrepare

With segmentation and mutation done, pass the result through {py:func}`~moleculekit.tools.preparation.systemPrepare` to assign protonation states:

```{code-cell} python
pmol, specs = systemPrepare(mut_min, verbose=False)
pmol.numAtoms
```

The full pipeline — segment, mutate, prepare — is now complete.

## Recap

- {py:func}`~moleculekit.tools.autosegment.autoSegment` detects backbone discontinuities from atomic coordinates and assigns a unique segid (and optionally chain letter) per backbone-continuous segment; use `fields=("chain", "segid")` to keep both fields consistent.
- {py:meth}`~moleculekit.molecule.Molecule.mutateResidue` with `sel` and `newres` swaps a residue's sidechain using Dunbrack rotamer selection: `rotamer_mode="best"` minimises VdW clashes against neighbours, `rotamer_mode="random"` samples by probability for speed. Add `minimize=True` to relax residual strain with OpenMM.
- {py:func}`~moleculekit.tools.modelling.model_gaps` fills missing residues by sequence using the ProMod3 loop-modelling engine — but it requires the ProMod3 Singularity image; there is no fallback.

## Next

- [System-preparation pipeline](../../explanation/system-preparation-pipeline.md)
- Back to the [system-prep index](index.md)
