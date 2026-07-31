from moleculekit.molecule import Molecule
from moleculekit.util import find_executable
from subprocess import run
import numpy as np
import tempfile
import logging
import os

logger = logging.getLogger(__name__)

CODE = """
from ost import io
from promod3 import modelling, loop

minimize = {minimize}
build_sidechains = {build_sidechains}

# setup
merge_distance = {merge_distance}
fragment_db = loop.LoadFragDB()
structure_db = loop.LoadStructureDB()
torsion_sampler = loop.LoadTorsionSamplerCoil()

# get raw model
tpl = io.LoadPDB("input.pdb")
aln = io.LoadAlignment("input.fasta")
aln.AttachView(1, tpl.CreateFullView())
mhandle = modelling.BuildRawModel(aln)

# we're not modelling termini
modelling.RemoveTerminalGaps(mhandle)

# perform loop modelling to close all gaps
modelling.CloseGaps(
    mhandle, merge_distance, fragment_db, structure_db, torsion_sampler
)

# build sidechains
if build_sidechains:
    modelling.BuildSidechains(
        mhandle, merge_distance, fragment_db, structure_db, torsion_sampler
    )

# minimize energy of final model using molecular mechanics
if minimize:
    modelling.MinimizeModelEnergy(mhandle)

# check final model and report issues
modelling.CheckFinalModel(mhandle)

# extract final model
final_model = mhandle.model
io.SavePDB(final_model, "model.pdb")"""


# Charged for a gap in the observed sequence at a position where the deposited
# backbone is covalently continuous, so no residue can be missing there. Far above
# any gap-open/mismatch trade the aligner could otherwise profit from (tens of
# points), yet finite: a full sequence that genuinely contains residues the backbone
# never had (an engineered internal deletion, or a break that went undetected) still
# aligns on sequence evidence instead of being forced into a nonsense frame shift.
_NO_BREAK_GAP_PENALTY = -1000.0


def _observed_backbone_breaks(mol, atom_idx, tol=2.0):
    """Observed-sequence positions where residues can be missing.

    Returns the set of observed-residue indices ``k`` that are not peptide-bonded to
    residue ``k-1``, i.e. where the deposited backbone is broken: their C-N distance
    exceeds ``tol`` Angstrom, or one of the two atoms is absent so continuity cannot
    be established. Anywhere else the chain is covalently continuous, which is proof
    that no residue is missing there.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The structure the observed sequence was read from.
    atom_idx : list of numpy.ndarray
        One atom-index array per observed residue of a single chain, in sequence
        order, as returned by ``Molecule.getSequence(return_idx=True)``.
    tol : float
        Maximum C-N distance (Angstrom) still counted as a peptide bond.
    """
    breaks = set()
    for k in range(1, len(atom_idx)):
        dist = _peptide_cn_distance(mol, atom_idx[k - 1], atom_idx[k])
        if dist is None or dist > tol:
            breaks.add(k)
    return breaks


def _peptide_cn_distance(mol, prev_atoms, cur_atoms):
    """Distance (Angstrom) from the C of one residue to the N of the next, on the
    first frame. None when either atom is absent, i.e. when continuity cannot be
    established from coordinates at all."""
    c = prev_atoms[mol.name[prev_atoms] == "C"]
    n = cur_atoms[mol.name[cur_atoms] == "N"]
    if not len(c) or not len(n):
        return None
    return float(np.linalg.norm(mol.coords[c[0], :, 0] - mol.coords[n[0], :, 0]))


def _residue_groups(mol, sel):
    """Atoms of ``sel`` grouped per residue in file order.

    Returns a list of ``((segid, chain, resid, insertion), atom_indices)``. Residue
    boundaries come from a change in any of those four fields between adjacent
    atoms, matching how the rest of this module walks a structure: the depositor's
    ATOM order is the chain order, and sorting by resid mis-orders insertion codes.
    """
    idx = np.where(mol.atomselect(sel))[0]
    if len(idx) == 0:
        return []
    keys = list(
        zip(
            mol.segid[idx].astype(str),
            mol.chain[idx].astype(str),
            mol.resid[idx].astype(int),
            mol.insertion[idx].astype(str),
        )
    )
    groups = []
    start = 0
    for k in range(1, len(idx) + 1):
        if k == len(idx) or keys[k] != keys[start]:
            groups.append((keys[start], idx[start:k]))
            start = k
    return groups


def detectBackboneBreaks(mol, sel="protein", tol=2.0):
    """Report every place where the protein backbone is broken.

    Two residues are compared when they are adjacent in file order **and** share a
    segid and a chain, so a chain or segment boundary is never mistaken for a
    break. This is the check to run on a built or prepared structure: a break that
    survives into it is an unmodelled gap (or one the builder capped), and no job
    status reports it.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The structure to check. Only the first frame is used.
    sel : str
        Atom selection to check. The default limits it to protein residues.
    tol : float
        Maximum C-N distance (Angstrom) still counted as a peptide bond. A real
        peptide bond is ~1.33 A.

    Returns
    -------
    breaks : list of dict
        One entry per break, in file order: ``{"segid", "chain", "after_resid",
        "after_insertion", "before_resid", "before_insertion", "distance"}``.
        ``after_*`` identifies the residue the break follows and ``before_*`` the
        one it precedes (the same convention as :func:`detectSequenceGaps`).
        ``distance`` is ``None`` when the C or N atom itself is missing, which is a
        break in the sense that continuity cannot be established -- see
        :func:`moleculekit.tools.backbone.check_backbone` to rebuild such atoms.

    Examples
    --------
    >>> from moleculekit.tools.modelling import detectBackboneBreaks
    >>> breaks = detectBackboneBreaks(mol)                       # doctest: +SKIP
    >>> [(b["after_resid"], b["before_resid"]) for b in breaks]  # doctest: +SKIP
    [(964, 977)]
    """
    groups = _residue_groups(mol, sel)
    breaks = []
    for (kprev, prev), (kcur, cur) in zip(groups, groups[1:]):
        if kprev[:2] != kcur[:2]:  # different chain or segment: not consecutive
            continue
        dist = _peptide_cn_distance(mol, prev, cur)
        if dist is not None and dist <= tol:
            continue
        breaks.append(
            {
                "segid": kprev[0],
                "chain": kprev[1],
                "after_resid": kprev[2],
                "after_insertion": kprev[3],
                "before_resid": kcur[2],
                "before_insertion": kcur[3],
                "distance": dist,
            }
        )
    return breaks


def _align_full_to_observed(full_seq, observed_seq, breaks=None):
    """Global BLOSUM62 alignment of the full sequence (reference) against the
    observed sequence. Returns ``(aligned_full, aligned_observed)``; the observed
    side carries ``-`` at positions missing from the structure.

    Parameters
    ----------
    full_seq : str
        The complete one-letter target sequence.
    observed_seq : str
        The one-letter sequence actually present in the structure.
    breaks : set of int, optional
        Observed-sequence positions where the deposited backbone is discontinuous,
        from :func:`_observed_backbone_breaks`. Residues can only be missing where
        the backbone is broken, so the observed side opens a gap only at one of these
        positions or past a terminus, and which of them a missing run belongs to is
        then settled by sequence identity. Passing ``None`` lets the alignment place
        its gaps wherever the sequence alone prefers, which for a structure with two
        nearby holes is routinely wrong: one merged gap saves a gap-open penalty
        (-11) and therefore outscores the two real ones as soon as the residues in
        between can slide onto same-or-similar reference residues, silently pairing
        observed residues with the wrong reference residues and placing the missing
        run where the backbone was never broken.

    Returns
    -------
    aligned_full : str
        ``full_seq`` with alignment gaps.
    aligned_observed : str
        ``observed_seq`` aligned to ``full_seq`` with ``-`` where residues are
        absent.
    """
    try:
        from Bio.Align import PairwiseAligner, substitution_matrices
    except ImportError:
        raise ImportError(
            "You need the biopython package for sequence alignment. Install it "
            "with `conda install biopython`."
        )
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11.0
    aligner.extend_gap_score = -1.0
    if breaks is not None:
        nobs = len(observed_seq)

        def _observed_gap_score(index, length):
            # `index` counts the observed residues to the left of the gap. A gap is
            # free wherever residues legitimately CAN be missing, i.e. at a backbone
            # break or past either terminus: the total number of missing residues is
            # fixed by the two sequence lengths, so charging for them only decides
            # which hole they are shovelled into, and the usual -11 decides that by
            # gap bookkeeping (fewest gaps) instead of by sequence identity.
            if index == 0 or index == nobs or index in breaks:
                return 0.0
            return _NO_BREAK_GAP_PENALTY

        # biopython >= 1.85 renamed query_gap_score (gaps in the observed sequence)
        # to deletion_score. Probe the class, since reading the property on the
        # instance raises once open/extend scores differ.
        attr = (
            "deletion_score"
            if hasattr(type(aligner), "deletion_score")
            else "query_gap_score"
        )
        setattr(aligner, attr, _observed_gap_score)
    aln = aligner.align(full_seq, observed_seq)[0]
    return str(aln[0]), str(aln[1])


def _sequence_identity(aln_full, aln_obs):
    """Fraction of the observed residues (non-gap in ``aln_obs``) that align to an
    identical residue in ``aln_full``. 1.0 when the observed sequence is an exact
    sub-sequence of the full sequence."""
    matches = observed = 0
    for cf, co in zip(aln_full, aln_obs):
        if co == "-":
            continue
        observed += 1
        if cf == co:
            matches += 1
    return matches / observed if observed else 0.0


def _resolve_donor_label(donor, dseqs, label):
    """Map a ``chain_map`` donor label to an actual donor protein chain id: the
    chain id itself when present, else the chain id of the protein atoms whose
    segid equals the label (aceboltz writes its query chain id in the segid
    column). Returns None when the label matches neither."""
    if label in dseqs:
        return label
    protein = donor.atomselect("protein")
    match = protein & (donor.segid == label)
    if match.any():
        chains = np.unique(donor.chain[match])
        if len(chains) == 1:
            return str(chains[0])
    return None


def _pair_donor_chains(orig, donor, chain_map=None, min_identity=0.95):
    """Pair each original protein chain with the best-matching donor protein chain
    by sequence identity. Returns ``(pairing, unpaired)`` where ``pairing`` maps
    ``{original_chain: donor_chain}`` and ``unpaired`` lists original chains with no
    donor chain at >= ``min_identity``. ``chain_map``, if given, pins
    ``{donor_label: original_chain}`` entries (the donor label resolved against
    donor chain ids, then segids); auto-pairing fills in the rest."""
    oseqs, oidx = orig.getSequence(
        dict_key="chain", return_idx=True, sel="protein", _logger=False
    )
    dseqs, _ = donor.getSequence(
        dict_key="chain", return_idx=True, sel="protein", _logger=False
    )

    pinned = {}
    for donor_label, orig_chain in (chain_map or {}).items():
        dchain = _resolve_donor_label(donor, dseqs, donor_label)
        if dchain is None:
            raise RuntimeError(
                f"chain_map pins donor '{donor_label}' -> original '{orig_chain}', "
                "but no donor chain has that chain id or segid."
            )
        pinned[orig_chain] = dchain

    # A chain_map whose original-chain target names no actual protein chain is a
    # typo: it is never consulted below, so warn rather than let it vanish silently.
    for orig_chain in pinned:
        if orig_chain not in oseqs:
            logger.warning(
                f"chain_map pins a donor to original chain '{orig_chain}', but the "
                "structure has no such protein chain; the override is ignored. "
                f"Original protein chains are {sorted(oseqs)}."
            )

    pairing, unpaired = {}, []
    for ochain, oseq in oseqs.items():
        if ochain in pinned:
            pairing[ochain] = pinned[ochain]
            continue
        best, best_id = None, 0.0
        breaks = _observed_backbone_breaks(orig, oidx[ochain])
        for dchain, dseq in dseqs.items():
            aln_full, aln_obs = _align_full_to_observed(dseq, oseq, breaks=breaks)
            idn = _sequence_identity(aln_full, aln_obs)
            if idn > best_id:
                best, best_id = dchain, idn
        if best is not None and best_id >= min_identity:
            pairing[ochain] = best
        else:
            unpaired.append(ochain)
    return pairing, unpaired


_ANCHOR_BB = ("N", "CA", "C")


def _fit_donor_to_orig(donor_pts, orig_pts):
    """Rigid (Kabsch) fit mapping donor points onto original points. Returns
    ``(fit, rmsd)`` where ``fit = (U, dc, oc)`` is applied with :func:`_apply_fit`
    and ``rmsd`` is the residual (Angstrom) after the optimal rotation. A large
    ``rmsd`` means the two point sets do not correspond as a rigid body - e.g. the
    donor's flanking residues were aligned to the wrong original residues."""
    from moleculekit.align import _pp_measure_fit

    donor_pts = np.asarray(donor_pts, dtype=np.float64)
    orig_pts = np.asarray(orig_pts, dtype=np.float64)
    dc = donor_pts.mean(axis=0)
    oc = orig_pts.mean(axis=0)
    U, rmsd = _pp_measure_fit(donor_pts - dc, orig_pts - oc)
    return (U, dc, oc), float(rmsd)


def _apply_fit(coords, fit):
    """Apply a ``(U, dc, oc)`` transform from :func:`_fit_donor_to_orig` to an
    ``(N, 3)`` coordinate array."""
    U, dc, oc = fit
    return (np.asarray(coords, dtype=np.float64) - dc) @ U.T + oc


def _anchor_pairs(slot, donor):
    """Return ``(orig_xyz, donor_xyz)`` lists of matched backbone-atom coordinates
    for one co-observed slot: each of N/CA/C present in BOTH the original residue
    (``slot["frag"]``) and its aligned donor residue (``slot["pred_atoms"]``)."""
    of = slot["frag"]
    datoms = slot["pred_atoms"]
    dnames = donor.name[datoms]
    dcoords = donor.coords[datoms, :, 0]
    orig_xyz, donor_xyz = [], []
    for nm in _ANCHOR_BB:
        oi = np.where(of.name == nm)[0]
        di = np.where(dnames == nm)[0]
        if len(oi) and len(di):
            orig_xyz.append(of.coords[oi[0], :, 0])
            donor_xyz.append(dcoords[di[0]])
    return orig_xyz, donor_xyz


def _slot_orig_bb(slot, name):
    """Coordinate of backbone atom ``name`` from a slot's ORIGINAL residue frag,
    or None when absent."""
    f = slot["frag"]
    a = np.where(f.name == name)[0]
    return f.coords[a[0], :, 0] if len(a) else None


def _slot_donor_bb(slot, name, donor, fit):
    """Coordinate of backbone atom ``name`` from a slot's DONOR residue, moved by
    ``fit`` (i.e. where the atom lands when placed). For a NEW slot the frag IS the
    (untransformed) donor residue; for a co-observed slot the donor coordinates are
    at ``pred_atoms``. Returns None when the atom / donor counterpart is absent."""
    if slot["new"]:
        f = slot["frag"]
        a = np.where(f.name == name)[0]
        return _apply_fit(f.coords[a[0], :, 0], fit) if len(a) else None
    datoms = slot.get("pred_atoms")
    if datoms is None:
        return None
    k = np.where(donor.name[datoms] == name)[0]
    if not len(k):
        return None
    return _apply_fit(donor.coords[datoms[k[0]], :, 0], fit)


def model_gaps(
    mol: Molecule,
    sequence: str,
    segid: str,
    promod_img: str,
    minimize: bool = False,
    build_sidechains: bool = True,
    merge_distance: float = 4,
) -> Molecule:
    """Closes residue gaps in a Molecule by sequence using ProMod3.
    Requires a ProMod3 Singularity image; see Notes.

    This method will also mutate any residues in the Molecule that do not
    match the input sequence.

    Parameters
    ----------
    mol : Molecule
        The molecule containing the segment to model.
    sequence : str
        The sequence to model.
    segid : str
        The segment ID of the segment to model.
    promod_img : str
        The path to the ProMod3 apptainer/singularity image. Follow the instructions at
        https://openstructure.org/promod3/3.4/container/singularity/ to obtain this image.
    minimize : bool
        Whether to minimize the model after building it.
    build_sidechains : bool
        Whether to build sidechains after building the model.
    merge_distance : float
        The distance to merge fragments at.

    Returns
    -------
    modeled_segment : Molecule
        The modeled segment.

    Notes
    -----
    This function requires a ProMod3 Singularity / Apptainer image.  Follow
    the instructions at https://openstructure.org/promod3/ to obtain the
    image, then pass the path to the downloaded ``.sif`` file as
    ``promod_img``.  The function executes the modelling script inside the
    container via ``singularity exec``, so Singularity or Apptainer must be
    available on ``$PATH``.

    Examples
    --------
    >>> from moleculekit.molecule import Molecule
    >>> from moleculekit.tools.modelling import model_gaps
    >>> mol = Molecule("5VQ6")  # doctest: +SKIP
    >>> sequence = "HMTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKSDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEK"  # doctest: +SKIP
    >>> res = model_gaps(mol, sequence, "0", "./promod.img")  # doctest: +SKIP
    """
    promod_img = os.path.abspath(promod_img)

    apptainer = find_executable("apptainer")
    if not apptainer:
        apptainer = find_executable("singularity")
    if not apptainer:
        raise RuntimeError(
            "Could not find apptainer or singularity. Please install one of them."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        pdbfile = os.path.join(tmpdir, "input.pdb")
        mol_seg = mol.copy(sel=f"segid {segid}")
        mol_seg.write(pdbfile)

        molseqs, molidx = mol.getSequence(dict_key="segid", return_idx=True)
        molseq = molseqs[segid]

        _, aligned_observed = _align_full_to_observed(
            sequence, molseq, breaks=_observed_backbone_breaks(mol, molidx[segid])
        )

        fastafile = os.path.join(tmpdir, "input.fasta")
        with open(fastafile, "w") as f:
            # Need to add gaps to sequence
            f.write(f">REFERENCE\n{sequence}\n")
            f.write(f">{segid}\n{aligned_observed}")

        runpy = os.path.join(tmpdir, "run.py")
        with open(runpy, "w") as f:
            f.write(
                CODE.format(
                    minimize=minimize,
                    build_sidechains=build_sidechains,
                    merge_distance=merge_distance,
                )
            )

        run([apptainer, "run", "--app", "OST", promod_img, runpy], cwd=tmpdir)
        outfile = os.path.join(tmpdir, "model.pdb")
        if not os.path.exists(outfile):
            raise RuntimeError("Model could not be generated. Please check input.")

        modeled_segment = Molecule(outfile)

    return modeled_segment


def _chain_is_canonical(mol, chain):
    from moleculekit.tools.nonstandard_residues import _CANONICAL_RESNAMES

    sel = (mol.chain == chain) & mol.atomselect("protein")
    return all(rn in _CANONICAL_RESNAMES for rn in np.unique(mol.resname[sel]))


def detectSequenceGaps(mol, sequences):
    """Detect missing-residue gaps per protein chain by aligning the observed
    sequence to the supplied full sequence.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The gapped structure.
    sequences : dict
        ``{chain: full_sequence}`` (e.g. from ``resolveFullSequences``).

    Returns
    -------
    gaps : list of dict
        Each ``{"chain", "after_resid", "before_resid", "missing_seq",
        "is_terminal"}``. ``after_resid`` is the observed resid immediately before
        the gap (``None`` for an N-terminal gap); ``before_resid`` is the observed
        resid immediately after it (``None`` for a C-terminal gap).
    skipped_ncaa_chains : list of str
        Protein chains skipped because they contain non-canonical residues.
    mismatches : list of dict
        Positions where a residue *is* present but is not the residue the
        reference has there -- engineered mutations, natural variants, or a
        reference belonging to a different construct. Each
        ``{"chain", "resid", "insertion", "reference", "observed", "ref_index"}``,
        where ``reference``/``observed`` are the one-letter codes and ``ref_index``
        is the 0-based position in ``sequences[chain]``, so
        ``sequences[chain][ref_index]`` can be patched to the observed residue to
        model the construct as crystallised rather than the wild type. Positions
        where either side is ``X`` (unknown or modified residue) are not reported:
        they are not evidence of a substitution.
    """
    obsseq, obsidx = mol.getSequence(
        dict_key="chain", return_idx=True, sel="protein", _logger=False
    )
    gaps = []
    skipped = []
    mismatches = []
    for chain, obs in obsseq.items():
        if chain not in sequences or not obs:
            continue
        if not _chain_is_canonical(mol, chain):
            skipped.append(chain)
            continue
        # observed resids/insertions, one per observed residue (order matches obsseq)
        resids = [int(mol.resid[atoms[0]]) for atoms in obsidx[chain]]
        insertions = [str(mol.insertion[atoms[0]]) for atoms in obsidx[chain]]
        aligned_full, aligned_obs = _align_full_to_observed(
            sequences[chain], obs, breaks=_observed_backbone_breaks(mol, obsidx[chain])
        )

        oi = 0  # pointer into observed residues
        fi = 0  # pointer into sequences[chain]
        run = ""  # current run of missing residues
        run_after = None  # observed resid before the run
        for cf, co in zip(aligned_full, aligned_obs):
            if co != "-":
                # flush any pending gap ending here
                if run:
                    gaps.append(
                        {
                            "chain": chain,
                            "after_resid": run_after,
                            "before_resid": resids[oi],
                            "missing_seq": run,
                            "is_terminal": run_after is None,
                        }
                    )
                    run = ""
                # the residue is present but is not the one the reference has here
                if cf != "-" and cf != co and "X" not in (cf, co):
                    mismatches.append(
                        {
                            "chain": chain,
                            "resid": resids[oi],
                            "insertion": insertions[oi],
                            "reference": cf,
                            "observed": co,
                            "ref_index": fi,
                        }
                    )
                run_after = resids[oi]
                oi += 1
            elif cf != "-":
                run += cf
            if cf != "-":
                fi += 1
        if run:  # trailing (C-terminal) gap
            gaps.append(
                {
                    "chain": chain,
                    "after_resid": run_after,
                    "before_resid": None,
                    "missing_seq": run,
                    "is_terminal": True,
                }
            )
    return gaps, skipped, mismatches


def prepareGapModellingInput(mol, sequences, gaps, outdir):
    """Write the FASTA + gapped template PDB that aceboltz ``gapmodel`` consumes.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The gapped structure.
    sequences : dict
        ``{chain: full_sequence}``.
    gaps : list of dict
        The subset of ``detectSequenceGaps`` gaps the user chose to model.
    outdir : str
        Directory to write ``input.fasta`` and ``template.pdb`` into.

    Returns
    -------
    fasta_path : str
    template_path : str
    chain_map : dict
        ``{predicted_chain_label: original_chain}``. aceboltz remaps template
        chains to ``"0","1",...`` in the order written here.
    """
    import os

    selected = set((g["chain"], g["after_resid"], g["missing_seq"]) for g in gaps)
    matched = set()
    obsseq, obsidx = mol.getSequence(
        dict_key="chain", return_idx=True, sel="protein", _logger=False
    )

    # Canonical protein chains we have a full sequence for. `obsseq` keys come from
    # np.unique (sorted), which fixes the chain order used for both the template
    # and the fasta records below.
    chains = [c for c in obsseq if _chain_is_canonical(mol, c) and c in sequences]

    # Template: only those chains, gapped, waters/non-protein dropped. Built one
    # chain at a time in `chains` order so the template's chain order matches the
    # fasta record order (aceboltz remaps template chains to "0","1",... by
    # appearance, which must line up with chain_map).
    if not chains:
        raise RuntimeError(
            "No canonical protein chain with a resolved full sequence to model."
        )
    protein = mol.atomselect("protein")
    template = mol.copy(sel=(protein & (mol.chain == chains[0])))
    for chain in chains[1:]:
        template = _concat(template, mol.copy(sel=(protein & (mol.chain == chain))))
    template_path = os.path.join(outdir, "template.pdb")
    template.write(template_path)

    records = []
    chain_map = {}
    for i, chain in enumerate(chains):
        chain_map[str(i)] = chain
        resids = [int(mol.resid[a[0]]) for a in obsidx[chain]]
        aligned_full, aligned_obs = _align_full_to_observed(
            sequences[chain],
            obsseq[chain],
            breaks=_observed_backbone_breaks(mol, obsidx[chain]),
        )
        desired = []
        oi = 0
        run = ""
        run_after = None
        for cf, co in zip(aligned_full, aligned_obs):
            if co != "-":
                if run:
                    key = (chain, run_after, run)
                    if key in selected:
                        desired.append(run)
                        matched.add(key)
                    run = ""
                desired.append(co)
                run_after = resids[oi]
                oi += 1
            elif cf != "-":
                run += cf
        if run:
            key = (chain, run_after, run)
            if key in selected:
                desired.append(run)
                matched.add(key)
        records.append("".join(desired))

    unmatched = selected - matched
    if unmatched:
        raise RuntimeError(
            "Could not locate the following selected gap(s) in the sequence "
            f"alignment: {sorted(unmatched)}"
        )

    fasta_path = os.path.join(outdir, "input.fasta")
    with open(fasta_path, "w") as f:
        for i, seq in enumerate(records):
            f.write(f">{i}\n{seq}\n")
    return fasta_path, template_path, chain_map


def _insertion_letters(after_code, upper_code):
    """Insertion codes available for residues inserted between two residues that share
    a resid.

    Codes start strictly after ``after_code`` (the preceding residue's own code, ``""``
    when it has none) and stop strictly before ``upper_code``, which is ``None`` when
    the following residue has a higher resid and therefore imposes no bound.
    """
    start = ord("A") if after_code == "" else ord(after_code) + 1
    stop = ord("Z") + 1 if upper_code is None else ord(upper_code)
    return [chr(c) for c in range(start, stop)]


def _number_new_run(run, before, after):
    """Assign ``resid``/``insertion`` to one run of consecutive new-residue slots,
    placed between its flanking original residues.

    Uses the natural integer gap between the flanking residues when there is room,
    otherwise insertion codes anchored on the preceding residue. Terminal runs count
    outward from the single flanking residue.

    Returns the number of resids by which every residue after this run must be shifted
    to make the numbering fit, which is 0 for all of the above.
    """
    k = len(run)
    if before is not None and after is not None:
        a, ai, b = before["resid"], before["insertion"], after["resid"]
        if (b - a - 1) >= k:  # room for plain integers a+1 .. a+k
            for m, s in enumerate(run):
                s["resid"], s["insertion"] = a + 1 + m, ""
            return 0
        # no integer room: insertion codes anchored on `before`, continuing after its
        # own code and stopping before `after`'s when the two share a resid.
        letters = _insertion_letters(ai, after["insertion"] if b == a else None)
        if k > len(letters):
            # too long for insertion codes: open integer room by shifting everything
            # after this run up. `room` is negative when the flanks share a resid.
            room = b - a - 1
            for m, s in enumerate(run):
                s["resid"], s["insertion"] = a + 1 + m, ""
            return k - room
        for m, s in enumerate(run):
            s["resid"], s["insertion"] = a, letters[m]
    elif before is not None:  # C-terminal tail: count up from the last residue
        a = before["resid"]
        for m, s in enumerate(run):
            s["resid"], s["insertion"] = a + 1 + m, ""
    elif after is not None:  # N-terminal tail: count up to the first residue
        b = after["resid"]
        for m, s in enumerate(run):
            s["resid"], s["insertion"] = b - k + m, ""
    else:  # whole chain is new (no original residues): number from 1
        for m, s in enumerate(run):
            s["resid"], s["insertion"] = m + 1, ""
    return 0


def _warn_renumbered(slots, deposited, chain):
    """Warn once about the original residues whose deposited numbering was changed to
    make room for an inserted run, collapsing equally-shifted neighbours into ranges."""
    moved = [
        (d, s["resid"])
        for s, d in zip(slots, deposited)
        if d is not None and s["resid"] != d
    ]
    if not moved:
        return
    ranges = []
    for dep, new in moved:
        shift = new - dep
        if ranges and ranges[-1][2] == shift and ranges[-1][1] == dep - 1:
            ranges[-1][1] = dep
        else:
            ranges.append([dep, dep, shift])
    parts = ", ".join(
        f"{a} -> {a + s} (+{s})" if a == b else f"{a}-{b} -> {a + s}-{b + s} (+{s})"
        for a, b, s in ranges
    )
    logger.warning(
        f"Chain '{chain}': {len(moved)} original residue(s) renumbered to make room "
        f"for inserted run(s) too long for insertion codes: {parts}. Every other "
        "original residue keeps its deposited numbering."
    )


def _number_new_residues(slots, chain):
    """Fill in ``resid``/``insertion`` for the newly inserted residues in ``slots``,
    preserving the original residues' deposited numbering. A run too long for both the
    integer gap and insertion codes shifts the following original residues up to make
    room, which is reported with a warning."""
    deposited = [None if s["new"] else s["resid"] for s in slots]
    n = len(slots)
    i = 0
    while i < n:
        if not slots[i]["new"]:
            i += 1
            continue
        j = i
        while j < n and slots[j]["new"]:
            j += 1
        shift = _number_new_run(
            slots[i:j],
            slots[i - 1] if i > 0 else None,
            slots[j] if j < n else None,
        )
        if shift:
            # only the originals: following new slots are numbered later, from their
            # already-shifted flanks.
            for s in slots[j:]:
                if not s["new"]:
                    s["resid"] += shift
        i = j
    _warn_renumbered(slots, deposited, chain)


def _resolve_nonprotein_collisions(result, chain, slots):
    """Move this chain's non-protein residues (ligands, metals, waters) off any
    ``(resid, insertion)`` that the chain's numbered protein residues now occupy.

    The non-protein atoms of a modelled chain are split off before numbering, so they do
    not move with a shift and a protein residue can land on one. Collisions are keyed on
    ``(chain, resid, insertion)`` ignoring segid, because that is the triple written to
    PDB and used by downstream preparation; a same-chain, different-segid duplicate is
    still ambiguous. Each colliding residue is moved above the chain's highest resid,
    keeping the relative order of the moved residues.
    """
    mask = result.chain == chain
    if not mask.any():
        return
    idx = np.where(mask)[0]
    resids = result.resid[idx].tolist()
    insertions = result.insertion[idx].tolist()
    segids = result.segid[idx].tolist()

    taken = {(s["resid"], s["insertion"]) for s in slots}
    # dict.fromkeys keeps file order, so moved residues keep their relative order
    colliding = list(
        dict.fromkeys(
            key for key in zip(resids, insertions, segids) if (key[0], key[1]) in taken
        )
    )
    if not colliding:
        return

    nextid = max(max(s["resid"] for s in slots), max(resids)) + 1
    moves = []
    for resid, insertion, segid in colliding:
        sel = idx[
            (result.resid[idx] == resid)
            & (result.insertion[idx] == insertion)
            & (result.segid[idx] == segid)
        ]
        moves.append(f"{result.resname[sel[0]]} {resid}{insertion} -> {nextid}")
        result.resid[sel] = nextid
        result.insertion[sel] = ""
        nextid += 1
    logger.warning(
        f"Chain '{chain}': the inserted residues' numbering collided with "
        f"{len(moves)} non-protein residue(s) of the same chain, which were moved "
        f"above the chain's last residue: {', '.join(moves)}."
    )


def _superpose_and_graft_runs(
    slots,
    donor,
    graft_flanks,
    anchor_width=4,
    max_anchor_rmsd=2.0,
    junction_tol=1.6,
):
    """Place each run of newly inserted (donor) residues into the original's frame,
    or SKIP it if it cannot be attached with a physical junction. Returns the slot
    list with skipped runs' new residues removed (the caller assembles from it).

    For each run the donor is locally superposed onto the co-observed residues
    flanking the run; the run's frags are transformed into place and, where needed,
    flank residues are grafted from the transformed donor to keep the graft->kept
    peptide junction continuous.

    Grafting is per-side and minimal. Each closed side starts with NO graft and is
    extended outward one residue at a time - up to ``graft_flanks`` residues - only
    while its graft->kept junction is still stretched (> ``junction_tol`` Angstrom).
    A side whose new residue already joins the kept backbone cleanly grafts nothing,
    so a well-matching flank is never overwritten with donor coordinates; a side
    that diverges (e.g. a different crystal at a flexible loop) walks the seam out
    to a residue where the two structures locally superpose. ``graft_flanks`` is
    thus the MAXIMUM flank depth per side (default 1).

    If a closed side's junction still cannot be closed within ``graft_flanks``, the
    run is SKIPPED: its residues are not inserted, the gap is left as-is, and a
    warning names the loop and advises re-running ``spliceMissingResidues`` on the
    result with a larger ``graft_flanks`` (which retries only the still-missing gaps
    - a deeper graft trades a continuous junction for possible sidechain clashes to
    relax downstream). ``graft_flanks=0`` disables grafting and skipping entirely:
    every run is inserted in the original's frame as-is (used to test numbering /
    when the caller will relax junctions itself). A run with no kept flank at all (a
    whole modelled chain) has no junction to satisfy and is kept as-is.

    Warns when an accepted run's anchor fit residual exceeds ``max_anchor_rmsd``
    (Angstrom): the donor's flanking residues do not superpose rigidly, meaning the
    local alignment paired them with the wrong residues (mis-aligned / wrong donor)."""
    n = len(slots)

    def _fit(dl, dr, i, j):
        left = slots[max(0, i - dl - anchor_width) : max(0, i - dl)]
        right = slots[j + dr : j + dr + anchor_width]
        anchor = [
            s for s in left + right if not s["new"] and s.get("pred_atoms") is not None
        ]
        op, dp = [], []
        for s in anchor:
            a, b = _anchor_pairs(s, donor)
            op += a
            dp += b
        if len(op) < 3:
            # too few local anchors (very short chain or adjacent gaps): widen to a
            # whole-chain fit over every co-observed residue.
            op, dp = [], []
            for s in slots:
                if not s["new"] and s.get("pred_atoms") is not None:
                    a, b = _anchor_pairs(s, donor)
                    op += a
                    dp += b
        if len(op) < 3:
            return None
        return _fit_donor_to_orig(dp, op)

    def _junction(kept_slot, graft_slot, kept_atom, graft_atom, fit):
        kc = _slot_orig_bb(kept_slot, kept_atom)
        gc = _slot_donor_bb(graft_slot, graft_atom, donor, fit)
        if kc is None or gc is None:
            return None
        return float(np.linalg.norm(kc - gc))

    def _chain_label(idx):
        f = slots[idx]["frag"]
        return str(f.chain[0]) if f.numAtoms else "?"

    skip = set()
    placed = set()

    def _in_orig_frame(k):
        """Whether slot ``k``'s frag coordinates live in the original's frame, i.e.
        whether they can be measured against. A kept residue always does (deposited,
        or grafted donor coordinates already transformed). A NEW residue only does
        once its own run has been placed: runs are walked left to right, so a new
        slot belonging to a later run still holds raw donor coordinates, and one
        belonging to a skipped run keeps them for good. Measuring a junction against
        those compares coordinate frames rather than atoms, which for a donor sitting
        far from the original reads as a huge distance and skips a perfectly
        attachable run."""
        return not slots[k]["new"] or k in placed

    i = 0
    while i < n:
        if not slots[i]["new"]:
            i += 1
            continue
        j = i
        while j < n and slots[j]["new"]:
            j += 1

        left_closed, right_closed = i > 0, j < n
        if not (left_closed or right_closed):
            # a whole modelled chain: no kept residue to anchor to, no junction to
            # satisfy. Keep the donor coordinates as-is.
            i = j
            continue

        if graft_flanks <= 0:
            # insert-all: place the run in the original's frame (fit on the immediate
            # flanks) but graft no flanks and never skip on a bad junction.
            fitres = _fit(0, 0, i, j)
            if fitres is not None:
                fit, _ = fitres
                for s in slots[i:j]:
                    s["frag"].coords[:, :, 0] = _apply_fit(
                        s["frag"].coords[:, :, 0], fit
                    )
            i = j
            continue

        # Grow each closed side's graft from 0 outward, only as far as needed to
        # close its graft->kept junction (never past graft_flanks). The seam runs
        # between the outermost donor-sourced residue (the first new residue when a
        # side has grafted nothing yet) and the adjacent kept residue.
        dl = dr = 0
        fitres = _fit(dl, dr, i, j)
        for _ in range(n):
            if fitres is None:
                break
            fit, _ = fitres
            grew = False
            # N-side seam: C(kept i-dl-1) -> N(outermost donor residue i-dl).
            if left_closed and dl < graft_flanks and i - dl - 1 >= 1:
                jd = _junction(slots[i - dl - 1], slots[i - dl], "C", "N", fit)
                cand = slots[i - dl - 1]
                if (
                    jd is not None
                    and jd > junction_tol
                    and not cand["new"]
                    and cand.get("pred_atoms") is not None
                ):
                    dl += 1
                    grew = True
            # C-side seam: C(outermost donor j+dr-1) -> N(kept j+dr).
            if right_closed and dr < graft_flanks and j + dr < n - 1:
                jd = _junction(slots[j + dr], slots[j + dr - 1], "N", "C", fit)
                cand = slots[j + dr]
                if (
                    jd is not None
                    and jd > junction_tol
                    and not cand["new"]
                    and cand.get("pred_atoms") is not None
                ):
                    dr += 1
                    grew = True
            if not grew:
                break
            fitres = _fit(dl, dr, i, j)

        # Decide whether the run's closed-side junction(s) are acceptable. A side
        # whose seam residue is a not-yet-placed new residue has no junction to
        # satisfy here: the graft ran out to the neighbouring run, so that seam is
        # donor-to-donor and gets validated when the neighbouring run is walked (its
        # own seam residue is the kept flank this run just grafted).
        bad = fitres is None
        if fitres is not None:
            fit, rmsd = fitres
            if left_closed and i - dl - 1 >= 0 and _in_orig_frame(i - dl - 1):
                jd = _junction(slots[i - dl - 1], slots[i - dl], "C", "N", fit)
                bad = bad or (jd is None or jd > junction_tol)
            if right_closed and j + dr < n and _in_orig_frame(j + dr):
                jd = _junction(slots[j + dr], slots[j + dr - 1], "N", "C", fit)
                bad = bad or (jd is None or jd > junction_tol)

        if bad:
            after = slots[i - 1].get("resid") if left_closed else None
            ch = _chain_label(i - 1) if left_closed else _chain_label(j)
            logger.warning(
                f"Skipping {j - i} modelled residue(s) in chain {ch}"
                + (f" after resid {after}" if after is not None else " (N-terminus)")
                + f": the donor could not be attached with a physical backbone "
                f"junction within graft_flanks={graft_flanks}. Re-run "
                "spliceMissingResidues on the result with a larger graft_flanks to "
                "graft deeper and retry the remaining gaps (a deeper graft may "
                "introduce sidechain clashes to relax downstream)."
            )
            skip.update(range(i, j))
            i = j
            continue

        assert fitres is not None  # guaranteed: bad would be True above otherwise
        fit, rmsd = fitres
        if rmsd > max_anchor_rmsd:
            near = slots[i - 1].get("resid") if left_closed else slots[j].get("resid")
            logger.warning(
                f"Poor superposition of a modelled run (backbone anchor RMSD "
                f"{rmsd:.1f} A"
                + (f" near resid {near}" if near is not None else "")
                + "): the donor's flanking residues do not align rigidly to the "
                "original. The inserted segment's placement may be unreliable - "
                "check that the donor is the same protein and correctly aligned."
            )
        for s in slots[i:j]:
            s["frag"].coords[:, :, 0] = _apply_fit(s["frag"].coords[:, :, 0], fit)
        placed.update(range(i, j))
        graft = slots[max(0, i - dl) : i] + slots[j : j + dr]
        for s in graft:
            if not s["new"] and s.get("pred_atoms") is not None:
                frag = donor.copy(sel=s["pred_atoms"])
                frag.coords[:, :, 0] = _apply_fit(frag.coords[:, :, 0], fit)
                s["frag"] = frag
        i = j

    return [s for k, s in enumerate(slots) if k not in skip]


def spliceMissingResidues(
    mol, donor, chain_map=None, graft_flanks=1, min_identity=0.95
):
    """Insert only the newly added residues from ``donor`` into ``mol``.

    All original atoms (protein + ligands/metals/cofactors) are kept at their
    deposited coordinates AND with their deposited residue numbering, except the
    ``graft_flanks`` residues on each side of a filled gap (see below). Each modelled
    chain is rebuilt as its original residues plus the residues present in
    ``donor`` but absent from the original; each inserted residue is numbered to
    fall between its flanking original residues, using the natural integer gap where
    there is room and insertion codes otherwise. A run too long for both (more residues
    than the free insertion codes can number) instead shifts the following residues of
    that chain up to open integer room, reported with a warning; residues before the
    run and all other chains keep their deposited numbering.

    Each original protein chain is paired with the best-matching donor protein
    chain by sequence identity (labels are ignored), so donor chain/segid
    relabelling (e.g. aceboltz's ``model.pdb``) does not need a ``chain_map``.

    Where the missing residues go is read off the original's backbone: a run is only
    inserted where the deposited chain is actually broken, never where it is
    covalently continuous. Two holes a few residues apart (common once incomplete
    residues have been dropped) would otherwise be merged into one by the sequence
    alignment, which pairs the residues between them with the wrong donor residues.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The original (gapped) structure.
    donor : :class:`Molecule <moleculekit.molecule.Molecule>`
        The structure supplying the missing residues (a different crystal form,
        a higher-resolution entry, or an aceboltz ``model.pdb``).
    chain_map : dict, optional
        ``{donor_label: original_chain}`` overrides for chains that auto-pairing
        would get wrong or leave unpaired; the donor label is resolved against
        donor chain ids, then segids. Auto-pairing fills in every chain not
        pinned here. Default ``None`` (pure auto-pairing).
    min_identity : float
        Minimum sequence identity (fraction of the original chain's observed
        residues that match the donor) required to auto-pair an original chain
        to a donor chain. Original chains with no donor chain meeting this
        threshold are left unmodelled and reported via a warning.
    graft_flanks : int
        Number of original residues on each side of an inserted run to also take
        from ``donor`` (default 1). A loop modeller rebuilds the backbone of the
        residues immediately flanking a gap so the new segment closes; keeping the
        original flanking residue instead would leave a stretched junction peptide
        bond, which downstream preparation reads as a chain break and caps. Grafting
        the flanking residues from ``donor`` (keeping their original numbering)
        restores a continuous backbone. Set to 0 to keep every original residue
        exactly.

    Returns
    -------
    spliced : :class:`Molecule <moleculekit.molecule.Molecule>`
        ``mol`` with the modelled residues inserted.
    new_mask : numpy.ndarray
        Boolean mask over ``spliced`` marking the inserted atoms.
    """
    MARK = -987654  # temporary beta marker to recover inserted atoms after rebuild
    # Read-only aliases: neither is mutated in place (all mutation happens on
    # `result` and on the per-residue `.copy(sel=...)` frags), so there is no need
    # to copy the (possibly large) inputs. Do not start mutating them in place -
    # the caller's `mol` / `donor` must stay pristine.
    orig = mol
    pred = donor

    pairing, unpaired = _pair_donor_chains(orig, pred, chain_map, min_identity)
    for ochain in unpaired:
        logger.warning(
            f"No donor chain matched original chain '{ochain}' at >= {min_identity} "
            "sequence identity; leaving that chain unmodelled."
        )

    modelled_orig_chains = list(pairing.keys())
    keep_mask = np.logical_not(
        np.isin(orig.chain, modelled_orig_chains) & orig.atomselect("protein")
    )
    result = orig.copy()
    result.filter(keep_mask, _logger=False)
    # `result` may now be empty (single all-protein modelled chain); `_concat`
    # handles the first append by starting from the rebuilt chain when empty.

    for orig_chain, donor_chain in pairing.items():
        oseq, oidx = orig.getSequence(
            dict_key="chain",
            return_idx=True,
            sel=f"chain '{orig_chain}' and protein",
            _logger=False,
        )
        pseq, pidx = pred.getSequence(
            dict_key="chain",
            return_idx=True,
            sel=f"chain '{donor_chain}' and protein",
            _logger=False,
        )
        o = oseq[orig_chain]
        p = pseq[donor_chain]
        # predicted (donor) is the "full" side; the original's backbone breaks are
        # the only places its residues can be missing from
        aln_p, aln_o = _align_full_to_observed(
            p, o, breaks=_observed_backbone_breaks(orig, oidx[orig_chain])
        )

        segid = str(orig.segid[oidx[orig_chain][0][0]])

        # Walk the alignment into ordered residue slots, tagging the new residues and
        # remembering each original residue's deposited (resid, insertion) plus, when
        # it aligns to a predicted residue, that predicted residue's atoms (so a
        # remodelled flank can be grafted below).
        slots = []
        pi = oi = 0
        for cp, co in zip(aln_p, aln_o):
            if co != "-":  # original residue: keep verbatim
                atoms = oidx[orig_chain][oi]
                slots.append(
                    {
                        "frag": orig.copy(sel=atoms),
                        "new": False,
                        "resid": int(orig.resid[atoms[0]]),
                        "insertion": str(orig.insertion[atoms[0]]),
                        "pred_atoms": pidx[donor_chain][pi] if cp != "-" else None,
                    }
                )
                oi += 1
                if cp != "-":
                    pi += 1
            elif cp != "-":  # new residue from predicted
                slots.append(
                    {"frag": pred.copy(sel=pidx[donor_chain][pi]), "new": True}
                )
                pi += 1

        # Take the residues flanking each inserted run from the model too, so the
        # junction backbone is continuous (the modeller moved those flanks to close
        # the gap; keeping the original ones leaves a stretched, cappable bond).
        # Place/graft each run, or drop runs that cannot be attached cleanly.
        slots = _superpose_and_graft_runs(slots, pred, graft_flanks)

        # Number the new residues, preserving the originals, then stamp and append.
        _number_new_residues(slots, orig_chain)
        _resolve_nonprotein_collisions(result, orig_chain, slots)
        new_chain = None
        for s in slots:
            frag = s["frag"]
            frag.chain[:] = orig_chain
            frag.segid[:] = segid
            frag.resid[:] = s["resid"]
            frag.insertion[:] = s["insertion"]
            if s["new"]:
                frag.beta[:] = MARK
            new_chain = frag if new_chain is None else _concat(new_chain, frag)
        if new_chain is not None:
            result = _concat(result, new_chain)

    new_mask = result.beta == MARK
    result.beta[new_mask] = 0.0
    return result, new_mask


def detectSplicedClashes(mol, new_mask, cutoff=2.0, targets="not protein"):
    """Report steric overlaps between the newly modelled residues and other atoms.

    aceboltz models from the protein template alone, so a modelled loop can be
    placed into an ion / water / ligand site. This checks the ``new_mask`` atoms
    against ``targets`` (default the non-protein components) within ``cutoff``.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The spliced structure.
    new_mask : numpy.ndarray
        Boolean mask over ``mol`` marking the newly modelled atoms.
    cutoff : float
        Heavy-atom distance (Angstrom) below which a pair is a clash.
    targets : str
        Atom selection for the atoms the new residues must not clash with.

    Returns
    -------
    clashes : list of dict
        ``{"new_chain", "new_resid", "target_resname", "target_resid",
        "target_segid", "min_distance"}``, one per clashing new-residue / target
        pair, empty if clean.
    """
    from moleculekit.kdtree import cKDTree

    heavy = mol.element != "H"
    new_idx = np.where(new_mask & heavy)[0]
    tgt_idx = np.where(mol.atomselect(targets) & np.logical_not(new_mask) & heavy)[0]
    if len(new_idx) == 0 or len(tgt_idx) == 0:
        return []

    new_xyz = mol.coords[new_idx, :, 0]
    tgt_xyz = mol.coords[tgt_idx, :, 0]
    tree = cKDTree(tgt_xyz)
    pairs = tree.query_ball_point(new_xyz, r=cutoff)

    seen = {}
    for ni, tlist in enumerate(pairs):
        for tj in tlist:
            na, ta = new_idx[ni], tgt_idx[tj]
            d = float(np.linalg.norm(mol.coords[na, :, 0] - mol.coords[ta, :, 0]))
            key = (
                int(mol.resid[na]),
                str(mol.chain[na]),
                int(mol.resid[ta]),
                str(mol.segid[ta]),
            )
            if key not in seen or d < seen[key]["min_distance"]:
                seen[key] = {
                    "new_chain": str(mol.chain[na]),
                    "new_resid": int(mol.resid[na]),
                    "target_resname": str(mol.resname[ta]),
                    "target_resid": int(mol.resid[ta]),
                    "target_segid": str(mol.segid[ta]),
                    "min_distance": d,
                }
    return list(seen.values())


def _concat(a, b):
    if a.numAtoms == 0:
        return b.copy()
    out = a.copy()
    out.append(b, collisions=False)
    return out
