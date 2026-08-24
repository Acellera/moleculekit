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

# Filling missing loops from a homologous structure

**You will learn:** how to fill unresolved (missing-residue) regions of a crystal structure by grafting the missing residues from a more complete donor structure of the same protein, using {py:func}`~moleculekit.tools.modelling.spliceMissingResidues`, and how to find and check those gaps with {py:func}`~moleculekit.rcsb.resolveFullSequences`, {py:func}`~moleculekit.tools.modelling.detectSequenceGaps`, {py:func}`~moleculekit.tools.modelling.detectBackboneBreaks`, and {py:func}`~moleculekit.tools.modelling.detectSplicedClashes`.

**Prerequisites:**
- The [Mutation, gap closing, and auto-segmentation](04-mutation-gap-closing-segmentation.md) tutorial.

## Setup

```{code-cell} python
from moleculekit.molecule import Molecule
from moleculekit.rcsb import resolveFullSequences
from moleculekit.tools.modelling import (
    detectSequenceGaps,
    detectBackboneBreaks,
    spliceMissingResidues,
    detectSplicedClashes,
)
```

```{code-cell} python
:tags: [remove-input]
from acellera_docs_theme.molstar import show3d
```

## Step 1: Load a structure with unresolved regions

We use **5VQ2**, a GTP/Mg-bound KRAS homodimer (chains A and B) whose crystal has several unresolved regions per chain. We load it as deposited, keeping its cofactors and crystallographic waters:

```{code-cell} python
mol = Molecule("5VQ2")
```

```{code-cell} python
:tags: [remove-input]
show3d(mol)
```

## Step 2: Resolve the full sequence and find the gaps

{py:func}`~moleculekit.rcsb.resolveFullSequences` returns the full deposited sequence for each protein chain (keyed by chain, each value a dict with `sequence`, `source`, and `identity`). {py:func}`~moleculekit.tools.modelling.detectSequenceGaps` takes a plain `{chain: sequence}` mapping, so we unwrap the `sequence` field first:

```{code-cell} python
resolved = resolveFullSequences(mol, "5VQ2")
sequences = {ch: v["sequence"] for ch, v in resolved.items()}

gaps, skipped, mismatches = detectSequenceGaps(mol, sequences)
for g in gaps:
    print(g)
print("skipped chains:", skipped, " mismatches:", mismatches)
```

Each gap dict gives the chain, the observed resid just before (`after_resid`) and after (`before_resid`) the gap, the missing subsequence (`missing_seq`), and whether it is a chain terminus (`is_terminal`). Chains A and B show the identical set of three gaps: a 3-residue internal loop between resid 33 and 37 (`PTI`), a 10-residue internal loop between resid 58 and 69 (`AGQEEYSAMR`, the switch-II region), and a single unresolved residue after resid 168 (`K`) with no observed residue following it, so `before_resid` is `None` and `is_terminal` is `True`.

The two other return values report what the alignment could *not* treat as a plain gap. `skipped` lists protein chains left out because they contain non-canonical residues, and `mismatches` lists positions where a residue is present but differs from the one the reference has there (an engineered mutation, a natural variant, or a reference from a different construct). Both are empty here: the reference came from the entry's own deposited sequence, so it agrees with the model residue for residue.

## Step 3: Graft the missing residues from a donor crystal

**4DSO** is the same protein in a more complete crystal. {py:func}`~moleculekit.tools.modelling.spliceMissingResidues` pairs each original chain to its best-matching donor chain by sequence identity and inserts only the residues present in the donor but missing from our structure. All original atoms keep their deposited coordinates except the `graft_flanks` residues on each side of a filled gap, which are taken from the donor to seat the insertion:

```{code-cell} python
donor = Molecule("4DSO")
sp1, new1 = spliceMissingResidues(mol, donor, graft_flanks=1)
```

With `graft_flanks=1` the loops whose flanking residues already agree between the two crystals splice cleanly, but the flexible switch-II / C-terminal region diverges too far for a single-residue graft and is **skipped with a warning** (look for a `Skipping ... after resid 168` message above). `spliceMissingResidues` returns the new {py:class}`~moleculekit.molecule.Molecule` and a boolean mask marking the newly added atoms.

## Step 4: Re-run with a deeper graft to fill the rest

Re-running on the *result* with a larger `graft_flanks` grafts deeper into each flank, which closes the region the minimal graft skipped:

```{code-cell} python
sp2, new2 = spliceMissingResidues(sp1, donor, graft_flanks=2)
```

```{code-cell} python
:tags: [remove-input]
show3d(sp2)
```

Both chains are now full length and continuous. The two GTP molecules and both Mg ions are carried through untouched:

```{code-cell} python
for ch in ("A", "B"):
    n = len({int(r) for r in sp2.resid[sp2.atomselect("protein") & (sp2.chain == ch)]})
    print(f"chain {ch} residues:", n)
print("GTP atoms:", int((sp2.resname == "GTP").sum()), " MG atoms:", int((sp2.resname == "MG").sum()))
```

## Step 5: Choosing which gaps to fill

By default a splice fills every run the donor can supply, which includes residues lying *beyond* either terminus of your structure. A donor crystal observed over a wider range therefore lengthens your chains, even when nothing was missing between them. When you want a specific span, name the gaps you want.

`gaps` takes the same `chain` / `after_resid` / `before_resid` entries that {py:func}`~moleculekit.tools.modelling.detectSequenceGaps` and {py:func}`~moleculekit.tools.modelling.detectBackboneBreaks` return, so you filter one of those lists and pass it in. {py:func}`~moleculekit.tools.modelling.detectBackboneBreaks` only ever compares residues that are adjacent in the file, so it cannot report a terminal tail: passing its output fills exactly the breaks your structure actually has and leaves both termini as deposited.

Call {py:func}`~moleculekit.tools.modelling.detectBackboneBreaks` before the [auto-segmentation](04-mutation-gap-closing-segmentation.md) step, not after: it never reports a break across a segid boundary, and auto-segmenting starts a new segment at each break, so on an already-segmented structure it always comes back empty. {py:func}`~moleculekit.tools.modelling.detectSequenceGaps` skips a chain containing non-canonical residues entirely, so that chain contributes no entries and stays unfilled by this filter.

Starting again from the deposited structure, this closes the two internal loops per chain and leaves the unresolved C-terminal residue 169 off:

```{code-cell} python
internal_only, _ = spliceMissingResidues(
    mol, donor, graft_flanks=2, gaps=detectBackboneBreaks(mol)
)

for ch in ("A", "B"):
    resids = internal_only.resid[internal_only.atomselect("protein")
                                 & (internal_only.chain == ch)]
    print(f"chain {ch}: {int(resids.min())}-{int(resids.max())}, "
          f"{len(set(int(r) for r in resids))} residues")
```

Both chains still end at their deposited last residue rather than being extended to the donor's. Re-running the break check confirms the two internal loops are now closed, since an empty list means no backbone break remains:

```{code-cell} python
print(detectBackboneBreaks(internal_only))
```

The still-missing C-terminal residue does not show up here even though it stays unfilled: a chain terminus is not a backbone break, so {py:func}`~moleculekit.tools.modelling.detectBackboneBreaks` has nothing to report about it either way.

To go the other way and request a terminus explicitly, write the entry yourself: `after_resid` is `None` for an N-terminal run and `before_resid` is `None` for a C-terminal one, which is exactly how {py:func}`~moleculekit.tools.modelling.detectSequenceGaps` reports the ones it flags `is_terminal`. Passing `gaps=[]` fills nothing.

## Step 6: Check the modelled residues for clashes

A grafted loop can land in a cofactor, ion, or water site. {py:func}`~moleculekit.tools.modelling.detectSplicedClashes` checks the newly added atoms (the mask returned by the splice) against the non-protein components, which here include the crystallographic waters:

```{code-cell} python
clashes = detectSplicedClashes(sp2, new2)
print(clashes)
```

Here the run reports four overlaps: the deeper graft placed the modelled C-terminal residues 169 and 172 (in both chains) within the 2 Angstrom cutoff of crystallographic waters. Each entry names the offending grafted residue and the atom it overlaps, with the minimum distance. Overlaps with waters are expected when loops are grafted into a structure that still carries its deposited solvent; they are resolved downstream by removing or re-placing the affected waters, or by a short minimization. This check is what surfaces them before you get that far. To restrict the check to cofactors and ions only, pass `targets="resname GTP MG"`.

The filled structure is now ready to hand to {py:func}`~moleculekit.tools.preparation.systemPrepare` (see the [Basic protonation](01-basic-protonation.md) tutorial); sidechain strain introduced by a deeper graft is left for downstream minimization.

## Recap

- {py:func}`~moleculekit.rcsb.resolveFullSequences` gets the full deposited sequence per chain; unwrap its `sequence` field before passing a `{chain: sequence}` map to {py:func}`~moleculekit.tools.modelling.detectSequenceGaps`.
- {py:func}`~moleculekit.tools.modelling.detectSequenceGaps` returns `(gaps, skipped_chains, mismatches)`; each gap locates a missing-residue run between two observed resids, while `mismatches` flags residues that are present but disagree with the reference sequence.
- {py:func}`~moleculekit.tools.modelling.spliceMissingResidues` grafts the residues missing from your structure out of a more complete donor of the same protein, returning `(mol, new_mask)`; increase `graft_flanks` and re-run to close regions a minimal graft skips.
- Pass `gaps=` to {py:func}`~moleculekit.tools.modelling.spliceMissingResidues` to choose which runs are filled; without it a donor observed over a wider range extends your chains past both termini. `gaps=detectBackboneBreaks(mol)` fills only the breaks your structure has.
- {py:func}`~moleculekit.tools.modelling.detectSplicedClashes` flags newly modelled atoms that overlap cofactors, ions, or ligands.

## Next

- [System-preparation pipeline](../../explanation/system-preparation-pipeline.md)
- Back to the [system-prep index](index.md)
