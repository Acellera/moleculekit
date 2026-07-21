from moleculekit.molecule import Molecule
from moleculekit.util import find_executable
from subprocess import run
import numpy as np
import tempfile
import os

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


def _align_full_to_observed(full_seq, observed_seq):
    """Global BLOSUM62 alignment of the full sequence (reference) against the
    observed sequence. Returns ``(aligned_full, aligned_observed)``; the observed
    side carries ``-`` at positions missing from the structure.

    Parameters
    ----------
    full_seq : str
        The complete one-letter target sequence.
    observed_seq : str
        The one-letter sequence actually present in the structure.

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
    aln = aligner.align(full_seq, observed_seq)[0]
    return str(aln[0]), str(aln[1])


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

        molseq = mol.getSequence(dict_key="segid")[segid]

        _, aligned_observed = _align_full_to_observed(sequence, molseq)

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
    """
    obsseq, obsidx = mol.getSequence(dict_key="chain", return_idx=True, sel="protein", _logger=False)
    gaps = []
    skipped = []
    for chain, obs in obsseq.items():
        if chain not in sequences or not obs:
            continue
        if not _chain_is_canonical(mol, chain):
            skipped.append(chain)
            continue
        # observed resids, one per observed residue (order matches obsseq)
        resids = [int(mol.resid[atoms[0]]) for atoms in obsidx[chain]]
        aligned_full, aligned_obs = _align_full_to_observed(sequences[chain], obs)

        oi = 0  # pointer into observed residues
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
                run_after = resids[oi]
                oi += 1
            elif cf != "-":
                run += cf
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
    return gaps, skipped


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
            sequences[chain], obsseq[chain]
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


def _insertion_letter(m):
    """Return the insertion code for the ``m``-th (0-based) inserted residue."""
    return chr(ord("A") + m)


def _number_new_run(run, before, after):
    """Assign ``resid``/``insertion`` to one run of consecutive new-residue slots,
    placed between its flanking original residues.

    Uses the natural integer gap between the flanking residues when there is room,
    otherwise insertion codes anchored on the preceding residue. Terminal runs count
    outward from the single flanking residue.
    """
    k = len(run)
    if before is not None and after is not None:
        a, ai, b = before["resid"], before["insertion"], after["resid"]
        if ai == "" and (b - a - 1) >= k:  # room for plain integers a+1 .. a+k
            for m, s in enumerate(run):
                s["resid"], s["insertion"] = a + 1 + m, ""
        else:  # no integer room: insertion codes anchored on `before`
            if k > 26:
                raise RuntimeError(
                    f"Cannot number {k} inserted residues between resid {a} and "
                    f"{b} with insertion codes (max 26)."
                )
            for m, s in enumerate(run):
                s["resid"], s["insertion"] = a, _insertion_letter(m)
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


def _number_new_residues(slots):
    """Fill in ``resid``/``insertion`` for the newly inserted residues in ``slots``,
    preserving the original residues' deposited numbering."""
    n = len(slots)
    i = 0
    while i < n:
        if not slots[i]["new"]:
            i += 1
            continue
        j = i
        while j < n and slots[j]["new"]:
            j += 1
        _number_new_run(
            slots[i:j],
            slots[i - 1] if i > 0 else None,
            slots[j] if j < n else None,
        )
        i = j


def _graft_run_flanks(slots, pred, k):
    """Replace the coordinates of the ``k`` original residues flanking each run of
    newly inserted residues with the modeller's version of the same residue, keeping
    the original numbering, so the backbone stays continuous across the junction."""
    if k <= 0:
        return
    n = len(slots)
    i = 0
    while i < n:
        if not slots[i]["new"]:
            i += 1
            continue
        j = i
        while j < n and slots[j]["new"]:
            j += 1
        for s in slots[max(0, i - k) : i] + slots[j : j + k]:
            if not s["new"] and s.get("pred_atoms") is not None:
                s["frag"] = pred.copy(sel=s["pred_atoms"])
        i = j


def spliceModelledResidues(mol, predicted, chain_map, graft_flanks=1):
    """Insert only the newly modelled residues from ``predicted`` into ``mol``.

    All original atoms (protein + ligands/metals/cofactors) are kept at their
    deposited coordinates AND with their deposited residue numbering, except the
    ``graft_flanks`` residues on each side of a filled gap (see below). Each modelled
    chain is rebuilt as its original residues plus the residues present in
    ``predicted`` but absent from the original; each inserted residue is numbered to
    fall between its flanking original residues, using the natural integer gap where
    there is room and insertion codes otherwise.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The original (gapped) structure.
    predicted : :class:`Molecule <moleculekit.molecule.Molecule>`
        The modelled structure (aceboltz ``model.pdb`` or, in tests, the full
        structure).
    chain_map : dict
        ``{predicted_chain: original_chain}``.
    graft_flanks : int
        Number of original residues on each side of an inserted run to also take
        from ``predicted`` (default 1). A loop modeller rebuilds the backbone of the
        residues immediately flanking a gap so the new segment closes; keeping the
        original flanking residue instead would leave a stretched junction peptide
        bond, which downstream preparation reads as a chain break and caps. Grafting
        the flanking residues from ``predicted`` (keeping their original numbering)
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
    orig = mol.copy()
    pred = predicted.copy()

    # Only the PROTEIN atoms of a modelled chain are rebuilt below; every other
    # atom (other chains, plus ligands/metals/cofactors/waters that share a chain
    # id with a modelled protein - the common PDB arrangement) is carried over
    # untouched at its deposited coordinates.
    modelled_orig_chains = list(set(chain_map.values()))
    keep_mask = np.logical_not(
        np.isin(orig.chain, modelled_orig_chains) & orig.atomselect("protein")
    )
    result = orig.copy()
    result.filter(keep_mask, _logger=False)
    # `result` may now be empty (single all-protein modelled chain); `_concat`
    # handles the first append by starting from the rebuilt chain when empty.

    for pred_chain, orig_chain in chain_map.items():
        oseq, oidx = orig.getSequence(
            dict_key="chain",
            return_idx=True,
            sel=f"chain '{orig_chain}' and protein",
            _logger=False,
        )
        # The chain_map key is the predicted chain label from
        # prepareGapModellingInput (the FASTA record index "0","1",...). aceboltz's
        # model.pdb carries that label in the SEGID column, not chainID: gapmodel's
        # minimize() round-trips through OpenMM's PDB writer, which relabels the
        # chainID column by index (0->A, 1->B, ...) and preserves the real chain id
        # in segid. Resolve the predicted residues by segid first, then fall back to
        # chain for models that keep the label in the chain column (hand-built
        # inputs / the synthetic test molecules).
        pseq, pidx = pred.getSequence(
            dict_key="segid",
            return_idx=True,
            sel=f"segid '{pred_chain}' and protein",
            _logger=False,
        )
        if pred_chain not in pseq:
            pseq, pidx = pred.getSequence(
                dict_key="chain",
                return_idx=True,
                sel=f"chain '{pred_chain}' and protein",
                _logger=False,
            )
        if orig_chain not in oseq or pred_chain not in pseq:
            raise RuntimeError(
                f"Could not align predicted chain '{pred_chain}' to original chain "
                f"'{orig_chain}': one of them has no protein sequence."
            )
        o = oseq[orig_chain]
        p = pseq[pred_chain]
        aln_p, aln_o = _align_full_to_observed(p, o)  # predicted is the "full" side

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
                        "pred_atoms": pidx[pred_chain][pi] if cp != "-" else None,
                    }
                )
                oi += 1
                if cp != "-":
                    pi += 1
            elif cp != "-":  # new residue from predicted
                slots.append({"frag": pred.copy(sel=pidx[pred_chain][pi]), "new": True})
                pi += 1

        # Take the residues flanking each inserted run from the model too, so the
        # junction backbone is continuous (the modeller moved those flanks to close
        # the gap; keeping the original ones leaves a stretched, cappable bond).
        _graft_run_flanks(slots, pred, graft_flanks)

        # Number the new residues, preserving the originals, then stamp and append.
        _number_new_residues(slots)
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


def detectModelledClashes(mol, new_mask, cutoff=2.0, targets="not protein"):
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
            key = (int(mol.resid[na]), str(mol.chain[na]), int(mol.resid[ta]), str(mol.segid[ta]))
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
