# The system-preparation pipeline

{py:func}`~moleculekit.tools.preparation.systemPrepare` is moleculekit's all-in-one function for taking a raw PDB
structure and producing a properly protonated, hydrogen-optimized molecule
ready for MD parameterization. This page builds a mental model of what happens
inside the function, why each step exists, and how the parameters map onto that
process. For step-by-step worked examples, see the system-preparation tutorials
linked at the bottom.

## The pipeline at a glance

```{mermaid}
flowchart TD
    A[Input Molecule] --> B[autoSegment]
    B --> C[Molecule.mutateResidue]
    C --> D[model_gaps]
    D --> E[Molecule.templateResidueFromSmiles]
    E --> F[detectNonStandardResidues]
    F --> G[systemPrepare]
    G --> H[Prepared Molecule + details DataFrame]
```

Every node above is an entry point you can call directly. Each step is optional — skip any that your input does not need. The internals of `systemPrepare` itself (rename, PDB2PQR, PROPKA, debump, bond restore, …) are covered step-by-step below; you do not invoke them directly.

## Step 1 — Detect non-standard residues

{py:func}`~moleculekit.tools.nonstandard_residues.detectNonStandardResidues` inspects `mol.bonds` (or guesses them by
distance if `mol.bonds` is empty) and walks every residue. It emits one
**spec** per residue that needs special handling:

- **ChainResidueSpec** — a non-canonical amino acid embedded in a polypeptide,
  or a canonical AA whose sidechain makes a non-peptide covalent bond. Examples:
  selenomethionine; an ASN-ND2 glycosylated with NAG; a TYR coordinating a
  heme iron.
- **CovalentLigandSpec** — a non-chain residue with exactly one non-peptide
  bond (single-anchor covalent inhibitor, NAG stem of a glycan).
- **ScaffoldSpec** — a non-chain residue with two or more non-peptide bonds
  (bicyclic-peptide scaffold, multi-anchor inhibitor).
- **LigandSpec** — a non-chain residue with no covalent bonds to the protein
  (free small-molecule, solvent molecule other than water, ion).

Note: plain Cys–Cys disulfides are NOT returned by {py:func}`~moleculekit.tools.nonstandard_residues.detectNonStandardResidues`.
They are handled internally by {py:func}`~moleculekit.tools.preparation.systemPrepare` in step 2: the participating
cysteines are renamed `CYX` and the disulfide bonds are preserved across the
PDB2PQR roundtrip. You don't need a spec to make this work.

Metal-coordination bonds involving standalone metal ions (e.g. a Ca²⁺ ion
coordinated by protein oxygens) are skipped — the protein residues are left
unmodified. Coordinations where the metal lives *inside* a cofactor (e.g. Fe
in a heme coordinated by a Cys-SG) are kept: the cofactor gets a
`CovalentLigandSpec` and the donating residue becomes a `ChainResidueSpec` so
its protonation state (Tyr-O⁻, Cys-S⁻) is handled correctly.

## Step 2 — Rename residues for force-field compatibility

Based on the specs from step 1, canonical AAs at non-peptide junctions are
renamed:

| Original | Renamed to | Reason |
|---|---|---|
| CYS (disulfide) | CYX | Thiol H absent; PDB2PQR needs this name |
| CYS (metal coord.) | XX# bucket | Custom prepi shared across identical junctions |
| HIS | HID / HIE / HIP | Tautomer / charge state from PROPKA |
| ASP (neutral) | ASH | After titration |
| GLU (neutral) | GLH | After titration |
| LYS (neutral) | LYN | After titration |
| TYR (negative) | TYM | After titration |
| ARG (neutral) | AR0 | After titration |

Non-canonical AAs (NCAAs) that have been pre-templated with
{py:meth}`~moleculekit.molecule.Molecule.templateResidueFromSmiles` are also handled here so that PDB2PQR's
templates apply cleanly to their atoms.

## Step 3 — Add hydrogens via PDB2PQR

PDB2PQR adds missing heavy atoms and all polar hydrogens using its internal
force-field templates. The result is a fully hydrogenated structure at the
default protonation for each residue's name.

Residues in `no_prot` are excluded from hydrogen addition — useful when a
known-good H-placement already exists and you want PDB2PQR to leave it alone.

## Step 4 — Predict pKa and titrate (optional)

When `titration=True` (the default), PROPKA estimates the pKa of every
titratable residue in the context of the folded structure. At the target `pH`,
each titratable group is set to its dominant protonation state:

- ASP/GLU: neutral if `pKa > pH`
- HIS: HID (Nδ), HIE (Nε), or HIP (both) depending on H-bond geometry and pKa
- LYS, ARG, TYR: neutral if `pKa < pH`

This step is skipped when `titration=False`. In that case, all residues keep
their default protonation state (standard charge at pH 7 for canonical AAs).

## Step 5 — Flip and debump

PDB2PQR flips the amide groups of Asn and Gln, and the imidazole of His, to
find the orientation that forms the best hydrogen-bond network. It then
debumps any steric clashes introduced by the added hydrogens.

Residues in `no_opt` are held fixed and not flipped. Use this for residues in
a metal site or a known crystal-water network where the flip would break the
geometry.

## Step 6 — Hold residues at non-peptidic bonds

Standard protein residues that sit at a covalent junction to a non-protein
partner — disulfides, glycosidic bonds, metal coordinations, stapled
sidechain bonds, covalently bound ligands, etc. — need special handling so
PDB2PQR/PROPKA do not disturb the linkage or over-protonate the junction.

When `hold_nonpeptidic_bonds=True` (the default), `systemPrepare` detects
these junctions and, for each affected residue, adds it to the internal
hold lists rather than restoring anything afterward:

- both partners are added to `no_opt` and `no_titr`, so their geometry is
  not flipped/debumped and they are not titrated (which could disturb the
  covalent bond);
- the non-protein partner is also added to `no_prot`, so PDB2PQR does not
  add hydrogens to it (it is templated separately);
- the protein junction residue is deliberately left out of `no_prot`, so
  PDB2PQR still hydrogenates it; the hydrogen displaced by the covalent
  bond is removed afterwards.

Setting `hold_nonpeptidic_bonds=False` skips this special handling and lets
PDB2PQR/PROPKA process the junction residues as if the non-peptide bond were
not there. This option is rarely needed and should be used with care.

## Step 7 — Restore formal charges and termini

After the PDB2PQR roundtrip, the formal charges on non-standard residues and
termini can be incorrect because PDB2PQR assigns charges via its own tables
rather than propagating the values from the input. `systemPrepare` recaptures
the formal charges from the input before the call and restores them on the
corresponding atoms in the output.

This ensures that, for example, an N-terminal ammonium (formal charge +1) and
a deprotonated Tyr-O⁻ in a metal site (formal charge -1) survive the pipeline
with the correct charges for downstream force-field parameterization.

## Step 8 — Restore missing sidechains (optional)

When `restore_missing_sidechains=True`, {py:func}`~moleculekit.tools.preparation.systemPrepare` uses moleculekit's
Dunbrack-rotamer mutator to template back any canonical residues whose entire
sidechain is absent from the input structure (e.g. truncated crystal structures
or homology models). This runs **before** the PDB2PQR call so the reconstructed
atoms participate in all downstream protonation and optimization steps.

This option is off by default because mutating residues is a significant
structural change that should be made consciously.

## How parameters map onto the pipeline

| Parameter | Step | Effect |
|---|---|---|
| `pH` | 4 | PROPKA target pH for titration |
| `titration=False` | 4 | Skip PROPKA; use default protonation |
| `force_protonation=[(sel, "HID"), ...]` | 4 | Patch specific residues after PROPKA |
| `no_opt` | 5 | Exclude residues from H-bond flip and debump |
| `no_prot` | 3 | Exclude residues from hydrogen addition |
| `no_titr` | 4 | Exclude residues from titration |
| `hold_nonpeptidic_bonds=True` | 6 | Capture and restore non-peptidic bonds |
| `restore_missing_sidechains=True` | 8 | Template back absent canonical sidechains |
| `hydrophobic_thickness` | 4 | Warn on buried titratable residues (membrane context) |
| `detect_specs` | 1 | Supply a pre-computed spec list; bypass auto-detection |
| `return_details=True` | — | Return per-residue pKa / protonation DataFrame |

## The larger pipeline

`systemPrepare` is one piece of a broader system-preparation workflow. Several
preparation steps must happen **before** calling it:

### autoSegment before systemPrepare

PDB2PQR requires non-empty, consistent segment identifiers. Plain PDB files
typically have empty `segid` fields. Run {py:func}`~moleculekit.tools.autosegment.autoSegment` first:

```python
from moleculekit.molecule import Molecule
from moleculekit.tools.autosegment import autoSegment
from moleculekit.tools.preparation import systemPrepare

mol = Molecule("structure.pdb")
mol_seg = autoSegment(mol)
mol_out, specs = systemPrepare(mol_seg)
```

### templateResidueFromSmiles before systemPrepare

Non-canonical residues (covalent ligands, modified AAs, NCAAs) need their bond
topology, element assignments, and formal charges established before
{py:func}`~moleculekit.tools.preparation.systemPrepare` runs. Call {py:meth}`~moleculekit.molecule.Molecule.templateResidueFromSmiles` with `sel`, `smiles`, and `addHs=True` for each such residue:

```python
lig_mask = mol.resname == "LIG"
mol.templateResidueFromSmiles(lig_mask, "CC(=O)Nc1ccc(O)cc1", addHs=True)
mol_out, specs = systemPrepare(mol)
```

{py:func}`~moleculekit.tools.preparation.systemPrepare` calls {py:func}`~moleculekit.tools.nonstandard_residues.detectNonStandardResidues` internally, so it will pick
up the ligand spec automatically.

### mutateResidue before systemPrepare

When you want to swap canonical residues (e.g. a point mutation), call
{py:meth}`~moleculekit.molecule.Molecule.mutateResidue` with `sel` and `newres` first. The mutator uses Dunbrack rotamers
to place the new sidechain, and {py:func}`~moleculekit.tools.preparation.systemPrepare` then protonates and optimizes
the mutated residue along with the rest of the protein.

### model_gaps before systemPrepare

Missing loops or residues in a crystal structure can be modelled with
{py:func}`~moleculekit.tools.modelling.model_gaps` (from `moleculekit.tools.modelling`), which uses the ProMod3
Singularity image. Run gap-filling before {py:func}`~moleculekit.tools.preparation.systemPrepare` so the modelled
residues are also protonated correctly.

## Passing a pre-computed spec list

{py:func}`~moleculekit.tools.nonstandard_residues.detectNonStandardResidues` is called internally every time {py:func}`~moleculekit.tools.preparation.systemPrepare`
runs unless you supply a spec list via `detect_specs`. This is useful when:

- You want to inspect or filter the spec list before preparation (e.g. to
  remove a spec for a ligand you want to leave unparameterized).
- You are running `systemPrepare` in a batch workflow and want to cache the
  spec list.

```python
from moleculekit.tools.nonstandard_residues import detectNonStandardResidues
from moleculekit.tools.preparation import systemPrepare

mol = Molecule("structure.pdb")
specs = detectNonStandardResidues(mol)

# Inspect specs, filter if needed
ligand_specs = [s for s in specs if s.resname == "LIG"]

mol_out, applied_specs = systemPrepare(mol, detect_specs=ligand_specs)
```

Pass `detect_specs=[]` to skip non-standard residue handling entirely (all
residues treated as canonical).

## Further reading

- Tutorial: [Basic protonation](../tutorials/system-prep/01-basic-protonation.md)
- Tutorial: [Non-standard residues](../tutorials/system-prep/02-non-standard-residues.md)
- Tutorial: [Custom residues from SMILES](../tutorials/system-prep/03-custom-residues-from-smiles.md)
- Tutorial: [Mutations, gap closing, and segmentation](../tutorials/system-prep/04-mutation-gap-closing-segmentation.md)
