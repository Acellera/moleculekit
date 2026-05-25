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

# Non-standard residues and covalent modifications

**You will learn:** how to detect non-standard residues, covalent modifications,
and free ligands in a structure, and how to pass that information into
{py:func}`~moleculekit.tools.preparation.systemPrepare` so it preserves the right bonds and renames residues correctly
for the force field.

**Prerequisites:**
- The [Basic protonation](01-basic-protonation.md) tutorial.

## Setup

```{code-cell} python
from moleculekit.molecule import Molecule
from moleculekit.tools.preparation import systemPrepare
from moleculekit.tools.nonstandard_residues import (
    detectNonStandardResidues,
    ChainResidueSpec,
    ScaffoldSpec,
    CovalentLigandSpec,
    LigandSpec,
)
```

## Step 1 — Detect non-standard residues on a representative structure

We use **1R1J**, a thermolysin-like protease that carries three N-glycosylation
sites (NAG sugars covalently attached to Asn residues) and a non-covalent
zinc-chelating inhibitor (OIR). This gives us examples of all the important
spec types in a single structure.

```{code-cell} python
mol = Molecule("1R1J")
specs = detectNonStandardResidues(mol)
print(specs)
```

{py:func}`~moleculekit.tools.nonstandard_residues.detectNonStandardResidues` does **not** mutate `mol` — it just walks the bond
graph and returns a list of spec objects ({py:class}`~moleculekit.tools.nonstandard_residues.ChainResidueSpec`, {py:class}`~moleculekit.tools.nonstandard_residues.CovalentLigandSpec`, {py:class}`~moleculekit.tools.nonstandard_residues.LigandSpec`, or {py:class}`~moleculekit.tools.nonstandard_residues.ScaffoldSpec`) describing every residue
that needs special handling.

> **Note:** Plain Cys–Cys disulfide bonds are **not** in this list —
> {py:func}`~moleculekit.tools.preparation.systemPrepare` handles those internally by renaming Cys to CYX.
> {py:func}`~moleculekit.tools.nonstandard_residues.detectNonStandardResidues` targets non-canonical residues, sidechain
> crosslinks such as N-glycosylation or isopeptide bonds, and covalent or free
> ligands.

## Step 2 — Walk through each spec subclass

### ChainResidueSpec — chain-resident residue needing special handling

A {py:class}`~moleculekit.tools.nonstandard_residues.ChainResidueSpec` is emitted for every residue that sits inside a polypeptide
chain and needs special parameterization. This includes:

- **Non-canonical amino acids** embedded in a peptide chain (no inter-residue
  non-peptide bond).
- **Canonical amino acids** whose sidechain is covalently bonded to something
  outside the peptide backbone — an Asn N-glycosylated by a sugar, a Glu–Lys
  isopeptide bond, a Cys thioether to a scaffold.

The 1R1J structure has three Asn residues each bonded to a NAG sugar at their
ND2 atom. The detector emits a {py:class}`~moleculekit.tools.nonstandard_residues.ChainResidueSpec` for each, proposing a shared
renamed resname so the parameterizer generates one set of AMBER parameters for
all three:

```{code-cell} python
chain_specs = [s for s in specs if isinstance(s, ChainResidueSpec)]
for s in chain_specs:
    print(
        f"resname={s.resname!r:4s}  chain={s.residue.chain!r}  "
        f"resid={s.residue.resid:<6}  anchor_atom={s.anchor_atom!r}  "
        f"new_resname={s.new_resname!r}"
    )
```

Each {py:class}`~moleculekit.tools.nonstandard_residues.ChainResidueSpec` exposes:

| Attribute | Meaning |
|-----------|---------|
| `resname` | Residue name in the input structure |
| `residue` | `UniqueResidueID` (chain / resid / segid / insertion) |
| `new_resname` | Name to rename to before parameterization (`None` = no rename needed) |
| `anchor_atom` | Atom involved in the non-peptide bond (`None` for plain chain NCAAs) |
| `is_n_term` / `is_c_term` | Whether this is at the N- or C-terminus of a chain |

**Canonical amino acids that participate in a non-peptide bond get renamed too** — the parameterizer needs different atom names and missing-H counts than the standard residue. A cross-residue covalent bond between two canonical amino acids therefore produces **two** {py:class}`~moleculekit.tools.nonstandard_residues.ChainResidueSpec` entries, one per side of the bond.

5VBL's bound peptide is cyclized through an isopeptide bond. Loading it and filtering for canonical amino-acid {py:class}`~moleculekit.tools.nonstandard_residues.ChainResidueSpec` entries surfaces exactly the two endpoints — each with its own `new_resname` and its own `anchor_atom`:

```{code-cell} python
mol_5vbl = Molecule("5VBL")
specs_5vbl = detectNonStandardResidues(mol_5vbl)

CANONICAL_AAS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
isopeptide_endpoints = [
    s for s in specs_5vbl
    if isinstance(s, ChainResidueSpec) and s.resname in CANONICAL_AAS
]
for s in isopeptide_endpoints:
    print(
        f"resname={s.resname!r:4s}  chain={s.residue.chain!r}  "
        f"resid={s.residue.resid:<6}  anchor_atom={s.anchor_atom!r}  "
        f"new_resname={s.new_resname!r}"
    )
```

Both partners have `new_resname` set; the unique names tell antechamber to build a separate prepi for each side.

### CovalentLigandSpec — single-anchor covalent ligand

A {py:class}`~moleculekit.tools.nonstandard_residues.CovalentLigandSpec` is emitted for a free (non-chain-resident) residue with
exactly one covalent bond to the rest of the structure. In 1R1J, the NAG
N-acetylglucosamine sugars each attach to one Asn via a single C1-ND2 glycosidic
bond:

```{code-cell} python
cov_specs = [s for s in specs if isinstance(s, CovalentLigandSpec)]
for s in cov_specs:
    print(
        f"resname={s.resname!r}  chain={s.residue.chain!r}  "
        f"resid={s.residue.resid}"
    )
```

{py:class}`~moleculekit.tools.nonstandard_residues.CovalentLigandSpec` has two public attributes: `resname` and `residue`.

### LigandSpec — free non-covalent ligand

A {py:class}`~moleculekit.tools.nonstandard_residues.LigandSpec` covers non-chain-resident residues with **no** covalent bonds to
any other residue. In 1R1J, the thiorphan-class inhibitor OIR coordinates the
active-site zinc ion via O19 and S26, but those are metal-coordination contacts
(not covalent bonds), so the detector correctly classifies it as a free ligand:

```{code-cell} python
lig_specs = [s for s in specs if isinstance(s, LigandSpec)]
for s in lig_specs:
    print(
        f"resname={s.resname!r}  chain={s.residue.chain!r}  "
        f"resid={s.residue.resid}"
    )
```

{py:class}`~moleculekit.tools.nonstandard_residues.LigandSpec` also has two public attributes: `resname` and `residue`.

### ScaffoldSpec — multi-anchor covalent scaffold

A {py:class}`~moleculekit.tools.nonstandard_residues.ScaffoldSpec` is emitted for a non-chain-resident residue with **two or more**
covalent bonds going out to the polypeptide chain — typical of bicyclic peptide
scaffolds or multi-anchor covalent inhibitors.

For a live example we load **8QFZ chain B**, a lasso-peptide scaffold (LFI)
thioether-bonded to three CYS residues:

```{code-cell} python
mol_8qfz = Molecule("8QFZ")
mol_8qfz.filter("chain B", _logger=False)

specs_8qfz = detectNonStandardResidues(mol_8qfz)
scaffold_specs = [s for s in specs_8qfz if isinstance(s, ScaffoldSpec)]
for s in scaffold_specs:
    print(f"resname={s.resname!r}  chain={s.residue.chain!r}  resid={s.residue.resid}")
```

The LFI scaffold appears as a {py:class}`~moleculekit.tools.nonstandard_residues.ScaffoldSpec` because it bonds covalently to three
chain-resident CYS residues. Each of those CYS residues appears as a
{py:class}`~moleculekit.tools.nonstandard_residues.ChainResidueSpec` with a unique auto-generated rename target, because they sit
at different chain positions (N-terminal, mid-chain, C-terminal) and therefore
carry different capping atoms in solution.

{py:class}`~moleculekit.tools.nonstandard_residues.ScaffoldSpec` has two public attributes: `resname` and `residue`.

## Step 3 — Apply specs through systemPrepare

Pass the spec list to {py:func}`~moleculekit.tools.preparation.systemPrepare` via `detect_specs=` to apply the proposed
renames and preserve the cross-residue bonds that protonation would otherwise
drop:

```{code-cell} python
pmol, applied_specs = systemPrepare(mol, detect_specs=specs, verbose=False)
```

`detect_specs=specs` tells {py:func}`~moleculekit.tools.preparation.systemPrepare` to rename force-field-relevant
residues (Asn → shared auto-name so antechamber builds one prepi) and preserve
the glycosidic C1-ND2 bonds that PDB2PQR's hydrogenation step would otherwise
sever. `pmol` is a new {py:class}`~moleculekit.molecule.Molecule`; `mol` is unchanged.

## Step 4 — Suppress a specific spec

You can filter the spec list before passing it in. For example, to skip
preparation of the covalent NAG sugars (perhaps you will handle them in a
separate glycan-parameterization step) you can drop all `CovalentLigandSpec`
entries:

```{code-cell} python
specs_no_nag = [s for s in specs if not isinstance(s, CovalentLigandSpec)]
pmol_no_nag, _ = systemPrepare(mol, detect_specs=specs_no_nag, verbose=False)
```

You can also filter on a spec's public attributes. For instance, to keep only
the ASN entries (dropping OIR and leaving NAG out too):

```{code-cell} python
specs_asn_only = [s for s in specs if s.resname == "ASN"]
pmol_asn, _ = systemPrepare(mol, detect_specs=specs_asn_only, verbose=False)
```

Any spec you remove is simply ignored by {py:func}`~moleculekit.tools.preparation.systemPrepare`; it uses only the
entries you provide.

## Recap

- {py:func}`~moleculekit.tools.nonstandard_residues.detectNonStandardResidues` enumerates non-standard residues and covalent
  modifications without mutating `mol`.
- Cys–Cys disulfides are **not** returned by it — {py:func}`~moleculekit.tools.preparation.systemPrepare` handles those
  internally.
- Four spec subclasses cover chain crosslinks ({py:class}`~moleculekit.tools.nonstandard_residues.ChainResidueSpec`), bicyclic
  scaffolds ({py:class}`~moleculekit.tools.nonstandard_residues.ScaffoldSpec`), covalent ligands ({py:class}`~moleculekit.tools.nonstandard_residues.CovalentLigandSpec`), and free
  ligands ({py:class}`~moleculekit.tools.nonstandard_residues.LigandSpec`).
- Pass the spec list (or a filtered subset) into {py:func}`~moleculekit.tools.preparation.systemPrepare` with `detect_specs=...`
  to control renaming and bond preservation.

## Next

- [Custom residues from SMILES](03-custom-residues-from-smiles.md)
- [System-preparation pipeline](../../explanation/system-preparation-pipeline.md)
