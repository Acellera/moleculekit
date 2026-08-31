"""Classify each protein chain terminus as a real biological end or a cut.

The builders cap every protein terminus by default, which is right for a
construct boundary or an unfilled break and wrong for a mature protein's own
terminus - a real terminus is zwitterionic, and neutralising it invents
chemistry. Telling the two apart needs evidence from outside the coordinate
file, and this module is where that evidence is combined:

* a **terminal gap** proves truncation on its own - the reference extends past
  the last modelled residue, whatever the reference is;
* otherwise the terminus is flush with the reference, and it is a real end only
  if it maps onto a UniProt mature-chain boundary (via the SIFTS entity mapping
  or a precursor trim offset) - but "flush" is only readable when the chain was
  actually analysed for gaps, so a chain gap detection skipped gets ``unknown``
  at both ends rather than a boundary comparison against an assumed alignment.

No network I/O happens here: the caller supplies the mature spans.
"""

import logging

import numpy as np

from moleculekit.residues import (
    CAP_RESIDUE_NAMES,
    MODIFIED_PROTEIN_RESIDUE_NAMES,
    PROTEIN_RESIDUE_NAMES_WITH_VARIANTS,
)

logger = logging.getLogger(__name__)

#: The caps that can be *requested*: the subset of
#: :data:`moleculekit.residues.CAP_RESIDUE_NAMES` a force field ships a template
#: for (htmd carries ACE, NME and NHE only), plus ``none`` to leave the terminus
#: charged on AMBER's ``N*``/``C*`` variants. Deliberately narrower than the
#: recognised set: ``NMA`` and ``NH2`` are real cap resnames that can appear in a
#: structure but cannot be built onto one.
CAP_VOCABULARY = ("none", "ACE", "NME", "NHE")

# Residue names a force field can build a backbone cap onto - the same sets
# htmd's defaultProteinCaps uses to decide the very same question.
_CAPPABLE_RESNAMES = PROTEIN_RESIDUE_NAMES_WITH_VARIANTS | MODIFIED_PROTEIN_RESIDUE_NAMES

# Existing caps are recognised from CAP_RESIDUE_NAMES. They never appear as a
# terminal residue in this module's view: they are excluded from sel="protein"
# (ACE has no N/CA, NME no C/CA), so the terminal residue we see is the amino acid
# *underneath* the cap. Detection has to look at the neighbouring residue instead -
# see _adjacent_cap.


def _adjacent_cap(mol, chain, atoms, before):
    """Is the residue immediately before/after these atoms an existing cap?

    htmd's ``defaultProteinCaps`` sees a segment's atoms unfiltered, so an
    already-capped terminus reads as ACE/NME there and is left alone. This module
    works from ``sel="protein"``, which drops cap residues entirely, so without
    this check an ACE-ALA-NME peptide gets ACE and NME proposed onto it a second
    time.
    """
    idx = int(atoms[0]) - 1 if before else int(atoms[-1]) + 1
    if idx < 0 or idx >= mol.numAtoms:
        return False
    if str(mol.chain[idx]) != str(chain):
        return False
    return str(mol.resname[idx]) in CAP_RESIDUE_NAMES

# Max C(last)-N(first) distance still counted as a head-to-tail closure. htmd
# recognises cyclic segments at 1.35 A but honours longer modelled closures
# (7BTI's phalloidin at ~1.47 A, microcystin's at ~1.37 A), so allow 1.5.
_CYCLIC_TOL = 1.5


def _chain_is_cyclic(mol, residue_idx):
    """True when the chain's last C is bonded-distance from its first N."""
    first, last = residue_idx[0], residue_idx[-1]
    n_idx = [i for i in first if mol.name[i] == "N"]
    c_idx = [i for i in last if mol.name[i] == "C"]
    if not n_idx or not c_idx:
        return False
    d = np.linalg.norm(mol.coords[c_idx[0], :, 0] - mol.coords[n_idx[0], :, 0])
    return bool(d <= _CYCLIC_TOL)


def _uniprot_position(end, reflen, meta):
    """Which UniProt entry covers this terminus, and at which position.

    Returns ``(accession, position)``, or ``(None, None)`` when nothing maps it.
    One entity can align to several UniProt entries - a receptor with a fused
    lysozyme aligns to both - so every reference row is searched, and the answer
    carries the accession that actually covered the terminus. In a chimera the
    two ends legitimately belong to different proteins.
    """
    pos = 1 if end == "N" else int(reflen)
    for ref in meta.get("uniprot_refs") or []:
        for reg in ref["aligned_regions"]:
            beg = reg["entity_beg_seq_id"]
            if beg <= pos <= beg + reg["length"] - 1:
                return ref["accession"], reg["ref_beg_seq_id"] + (pos - beg)
    if meta.get("source") == "uniprot" and meta.get("accession"):
        # The reference is the precursor itself, trimmed by this many residues.
        return meta["accession"], pos + int(meta.get("trim_offset") or 0)
    return None, None


def _classify(end, accession, upos, mature_spans):
    """(classification, evidence, matched_feature) for one terminus."""
    spans = (mature_spans.get(accession) or []) if accession else []
    if upos is None or not spans:
        return "unknown", "flush_no_evidence", None
    key = "start" if end == "N" else "end"
    matching = [span for span in spans if span[key] == upos]
    if not matching:
        return "truncated", "uniprot_mature_chain", None
    # Several spans can share a boundary: P00760 starts the mature chain (24-246)
    # and alpha-trypsin chain 1 (24-148) at the same residue. Report the longest,
    # which is the physiological chain rather than an autolysis fragment.
    span = max(matching, key=lambda s: s["end"] - s["start"] + 1)
    feature = f"{span['type']} {span['start']}-{span['end']}"
    if span["description"]:
        feature = f"{feature} {span['description']}"
    return "natural", "uniprot_mature_chain", feature


def _unique_selection(mol, chain, resid, insertion, segid, resname):
    """The shortest selection string that resolves to exactly this residue.

    The chain is quoted because it is routinely blank: a structure with no chain
    identifiers turns an unquoted ``chain  and resid 2`` into a parse error, which
    the escalation ladder cannot recover from since every candidate is malformed
    the same way. ``segid`` gets its own rung because ``UniqueResidueID`` requires
    segid to be single-valued too, so a chain letter reused across two segments
    needs it to disambiguate.
    """
    from moleculekit.molecule import UniqueResidueID

    base = f'chain "{chain}" and resid {resid}'
    candidates = (
        base,
        f'{base} and insertion "{insertion}"',
        f'{base} and insertion "{insertion}" and segid "{segid}"',
        f'{base} and insertion "{insertion}" and segid "{segid}" and resname {resname}',
    )
    for sel in candidates:
        try:
            UniqueResidueID.fromMolecule(mol, sel)
        except RuntimeError:
            continue
        return sel
    logger.warning(
        f"Could not build a unique atom selection for chain {chain} resid "
        f"{resid}{insertion} ({resname}). Its cap cannot be requested."
    )
    return None


def _segments(mol, residue_idx, chain_gaps):
    """Runs of residues with no unmodelled gap between them.

    A chain is one segment only when every gap in it was filled. Leave an internal
    gap unmodelled and the built system has two pieces, each with its own pair of
    ends -- 1LV1's GGSSG linker is never modelled, so its single chain arrives at
    the builder as two protease copies with four termini between them, and the
    builder caps whichever it is not told about.
    """
    breaks = {
        (g["after_resid"], g["before_resid"])
        for g in chain_gaps
        if g.get("after_resid") is not None and g.get("before_resid") is not None
    }
    if not breaks:
        return [list(residue_idx)]
    segments, current = [], [residue_idx[0]]
    for previous, nxt in zip(residue_idx, residue_idx[1:]):
        pair = (int(mol.resid[previous[0]]), int(mol.resid[nxt[0]]))
        if pair in breaks:
            segments.append(current)
            current = [nxt]
        else:
            current.append(nxt)
    segments.append(current)
    return segments


def detectTermini(mol, sequences, gaps, chainmeta, mature_spans, skipped_chains=()):
    """Classify both ends of every protein chain as natural, truncated or unknown.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The structure the caps will be applied to. Must be the final, gap-filled
        structure: a terminus that exists only because residues are missing is
        not the terminus the build will have.
    sequences : dict
        ``{chain: reference_sequence}`` as used for gap detection.
    gaps : list of dict
        ``detectSequenceGaps`` output for the same structure and references.
    chainmeta : dict
        ``{chain: {"source", "accession", "uniprot_refs", "trim_offset"}}`` - the
        survey's per-chain reference metadata. ``uniprot_refs`` is every UniProt
        row with its SIFTS regions; ``accession`` is the primary (most-covered)
        one, used only as a fallback label.
    mature_spans : dict
        ``{accession: uniprotMatureChains(accession)}`` for every accession in
        ``uniprot_refs``, fetched by the caller.
    skipped_chains : iterable of str
        Chains whose gap analysis was skipped - ``detectSequenceGaps``' second
        return value. Flushness is read from the absence of a terminal gap, so a
        chain with no gap analysis has no flushness evidence at all and its
        termini are reported ``unknown`` rather than measured against the
        reference's ends.

    Returns
    -------
    termini : list of dict
        Two entries per protein chain (``end`` ``"N"`` then ``"C"``), each with
        ``chain``, ``end``, ``resid``, ``insertion``, ``resname``, ``sel``,
        ``classification`` (``natural`` / ``truncated`` / ``unknown``),
        ``evidence`` (``uniprot_mature_chain`` / ``terminal_gap`` /
        ``no_gap_analysis`` / ``flush_no_evidence``), ``accession``,
        ``matched_feature``, ``cappable`` and ``proposed_cap``. ``sel`` is a
        verified-unique atom selection for the residue, and is None when the
        terminus cannot be capped or cannot be selected unambiguously.
    """
    obsseq, obsidx = mol.getSequence(
        dict_key="chain", return_idx=True, sel="protein", _logger=False
    )
    out = []
    for chain in sorted(obsseq):
        if not obsseq[chain]:
            continue
        residue_idx = obsidx[chain]
        cyclic = _chain_is_cyclic(mol, residue_idx)
        meta = chainmeta.get(chain) or {}
        ref = sequences.get(chain)
        chain_gaps = [g for g in gaps if g["chain"] == chain]
        no_gap_analysis = str(chain) in {str(c) for c in skipped_chains}
        # Per segment, not per chain: an unmodelled internal gap makes two of them.
        segments = _segments(mol, residue_idx, chain_gaps)
        ends = []
        for index, segment in enumerate(segments):
            first, last = index == 0, index == len(segments) - 1
            ends.append((
                "N", segment[0],
                first and any(g["after_resid"] is None for g in chain_gaps),
                not first,
            ))
            ends.append((
                "C", segment[-1],
                last and any(g["before_resid"] is None for g in chain_gaps),
                not last,
            ))
        for end, atoms, has_terminal_gap, at_break in ends:
            a = atoms[0]
            resid, insertion = int(mol.resid[a]), str(mol.insertion[a])
            resname = str(mol.resname[a])
            accession = meta.get("accession")

            if at_break:
                # An end the structure has because residues either side of it were
                # never modelled. It is a cut by construction -- no reference can
                # make it a biological terminus -- so it needs a cap.
                classification, evidence, feature = "truncated", "internal_gap", None
            elif has_terminal_gap:
                classification, evidence, feature = "truncated", "terminal_gap", None
            elif no_gap_analysis:
                # No gap list for this chain, so "flush" is an assumption rather
                # than a finding: the observed sequence may stop well short of the
                # reference with nothing to say so.
                classification, evidence, feature = "unknown", "no_gap_analysis", None
            elif ref is None:
                classification, evidence, feature = (
                    "unknown",
                    "flush_no_evidence",
                    None,
                )
            else:
                covering, upos = _uniprot_position(end, len(ref), meta)
                classification, evidence, feature = _classify(
                    end, covering, upos, mature_spans
                )
                if covering is not None:
                    accession = covering

            # Resolve the selection FIRST: a terminus we cannot name unambiguously
            # cannot be capped, whatever its chemistry, and reporting
            # cappable=True with sel=None would hand the next task {None: "ACE"}.
            segid = str(mol.segid[a])
            sel = _unique_selection(mol, chain, resid, insertion, segid, resname)
            cappable = (
                not cyclic
                and resname in _CAPPABLE_RESNAMES
                and not _adjacent_cap(mol, chain, atoms, before=(end == "N"))
                and sel is not None
            )
            proposed = None
            if cappable:
                if classification == "natural":
                    proposed = "none"
                elif classification == "truncated":
                    proposed = "ACE" if end == "N" else "NME"

            out.append(
                {
                    "chain": str(chain),
                    "end": end,
                    "resid": resid,
                    "insertion": insertion,
                    "resname": resname,
                    "sel": sel if cappable else None,
                    "classification": classification,
                    "evidence": evidence,
                    "accession": accession,
                    "matched_feature": feature,
                    "cappable": cappable,
                    "proposed_cap": proposed,
                }
            )
    return out
