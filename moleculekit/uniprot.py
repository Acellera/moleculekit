# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
"""Sequence lookups against the UniProt REST API.

Use this when the RCSB cannot give the full target sequence of a chain --
:func:`moleculekit.rcsb.resolveFullSequences` has no entry for it, or the
structure is not a PDB deposition at all -- and a reference sequence is still
needed to tell which residues a structure is missing.

UniProt entries are **full-length precursors**: they include signal peptides,
propeptides and, for engineered constructs, sequence the crystallised protein
never had. Passing one straight to
:func:`moleculekit.tools.modelling.detectSequenceGaps` therefore reports a long
terminal "gap" that is not missing density but a region excised by design, so
trim it to the observed span with :func:`trimPrecursorSequences` first.
"""

import json
import logging
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

_UNIPROT_REST = "https://rest.uniprot.org/uniprotkb"


def _getUniProtJson(url, attempts=3):
    import time

    last_err = None
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as err:
            # 400/404 etc. - the query or accession is wrong, retrying won't help
            raise RuntimeError(f"UniProt request failed for {url}: {err}") from err
        except Exception as err:
            last_err = err
            logger.warning(
                f"Failed to connect to URL {url} with error {err}. Sleeping 5s and retrying."
            )
            time.sleep(5)
    raise RuntimeError(f"Failed to connect to URL {url}: {last_err}")


def uniprotSearch(query, size: int = 5) -> list:
    """Search UniProtKB and return the top hits as plain dicts.

    Parameters
    ----------
    query : str
        A UniProtKB query, e.g. ``'trypsin AND organism_name:"Bos taurus"'`` or a
        gene name. Field syntax is documented at
        https://www.uniprot.org/help/query-fields.
    size : int
        Maximum number of hits to return.

    Returns
    -------
    hits : list of dict
        ``{"accession", "name", "organism", "length", "reviewed"}`` per hit, best
        match first. ``reviewed`` marks a Swiss-Prot (manually curated) entry.
        Ranking is UniProt's own relevance, which is not sequence identity to any
        structure -- present the candidates and let the user pick rather than
        taking the first hit.

    Examples
    --------
    >>> from moleculekit.uniprot import uniprotSearch
    >>> hits = uniprotSearch('trypsin AND organism_name:"Bos taurus"')  # doctest: +SKIP
    >>> hits[0]["accession"]                                            # doctest: +SKIP
    'P00760'
    """
    fields = "accession,protein_name,organism_name,length,reviewed"
    url = (
        f"{_UNIPROT_REST}/search?query={urllib.parse.quote(str(query))}"
        f"&fields={fields}&format=json&size={int(size)}"
    )
    hits = []
    for entry in _getUniProtJson(url).get("results", []):
        desc = entry.get("proteinDescription", {}) or {}
        name = (
            desc.get("recommendedName", {}).get("fullName", {}).get("value")
            if desc.get("recommendedName")
            else None
        )
        if name is None and desc.get("submissionNames"):
            name = desc["submissionNames"][0].get("fullName", {}).get("value")
        hits.append(
            {
                "accession": entry.get("primaryAccession"),
                "name": name,
                "organism": (entry.get("organism", {}) or {}).get("scientificName"),
                "length": (entry.get("sequence", {}) or {}).get("length"),
                "reviewed": "reviewed" in (entry.get("entryType") or "").lower(),
            }
        )
    return hits


def uniprotSequence(accession: str) -> str:
    """The full-length precursor sequence of a UniProt entry.

    Parameters
    ----------
    accession : str
        A UniProtKB accession, e.g. ``"P00760"``.

    Returns
    -------
    sequence : str
        The one-letter sequence, signal peptide and propeptides included. Trim it
        to a structure with :func:`trimPrecursorSequences` before using it as a
        gap-detection reference.

    Examples
    --------
    >>> from moleculekit.uniprot import uniprotSequence
    >>> len(uniprotSequence("P00760"))   # doctest: +SKIP
    246
    """
    url = f"{_UNIPROT_REST}/{urllib.parse.quote(str(accession))}.json?fields=sequence"
    data = _getUniProtJson(url)
    try:
        return data["sequence"]["value"]
    except (KeyError, TypeError) as err:
        raise RuntimeError(
            f"UniProt entry {accession!r} returned no sequence"
        ) from err


_MATURE_FEATURES = ("Chain", "Peptide")
_LEADER_FEATURES = ("Signal", "Transit peptide", "Propeptide")


def uniprotMatureChains(accession: str) -> list:
    """The mature chain spans of a UniProt entry, in precursor numbering.

    A structure's terminus is a real biological end only if it coincides with a
    boundary of one of these spans: everything else is a cut through the
    backbone. Entries can carry several spans (bovine trypsin P00760 lists the
    mature chain 24-246 *and* its two alpha-trypsin autolysis products), so all
    of them are returned and the caller records which one matched.

    Parameters
    ----------
    accession : str
        A UniProtKB accession, e.g. ``"P00760"``.

    Returns
    -------
    spans : list of dict
        ``{"start", "end", "type", "description"}`` per span, in precursor
        numbering (1-based, inclusive), sorted by ``start`` then ``end``. When
        the entry declares no ``Chain``/``Peptide`` feature, a single span is
        synthesised from the precursor minus any leading signal, transit or
        propeptide (``type="synthesised"``).

    Examples
    --------
    >>> from moleculekit.uniprot import uniprotMatureChains
    >>> uniprotMatureChains("P00533")[0]["start"]    # doctest: +SKIP
    25
    """
    fields = "ft_signal,ft_propep,ft_chain,ft_transit,ft_peptide,sequence"
    url = (
        f"{_UNIPROT_REST}/{urllib.parse.quote(str(accession))}.json?fields={fields}"
    )
    data = _getUniProtJson(url)
    length = (data.get("sequence") or {}).get("length")
    if not length:
        raise RuntimeError(f"UniProt entry {accession!r} returned no sequence length")

    features = data.get("features") or []
    spans = []
    for feat in features:
        if feat.get("type") not in _MATURE_FEATURES:
            continue
        loc = feat.get("location") or {}
        start = (loc.get("start") or {}).get("value")
        end = (loc.get("end") or {}).get("value")
        if start is None or end is None:  # an unknown-position feature is no evidence
            continue
        spans.append(
            {
                "start": int(start),
                "end": int(end),
                "type": str(feat["type"]),
                "description": str(feat.get("description") or ""),
            }
        )

    if not spans:
        # Cleaved pieces, as (start, end, type). Only the ones that form a
        # contiguous run from residue 1, or one ending at the precursor's last
        # residue, are removable: a propeptide in the middle is not evidence
        # about either terminus.
        cleaved = []
        for feat in features:
            if feat.get("type") not in _LEADER_FEATURES:
                continue
            loc = feat.get("location") or {}
            start = (loc.get("start") or {}).get("value")
            end = (loc.get("end") or {}).get("value")
            if start is None or end is None:
                continue
            cleaved.append((int(start), int(end), str(feat["type"])))
        cleaved.sort()

        mature_start, used = 1, []
        for start, end, kind in cleaved:  # contiguous run from the N-terminus
            if start != mature_start:
                break
            mature_start = end + 1
            used.append(kind)

        mature_end = int(length)
        for start, end, kind in reversed(cleaved):  # ... and from the C-terminus
            if end != mature_end:
                break
            mature_end = start - 1
            used.append(kind)

        if mature_start > mature_end:
            # The whole precursor is cleaved away (P13948 is a 21-residue
            # propeptide and nothing else), so there is no mature chain to
            # report. Returning nothing lets the caller degrade to "unknown"
            # instead of matching against an impossible span.
            return []

        names = sorted(set(used))
        desc = "precursor minus " + ", ".join(names) if names else "precursor"
        spans.append(
            {
                "start": mature_start,
                "end": mature_end,
                "type": "synthesised",
                "description": desc,
            }
        )

    # A span must be a real 1-based inclusive range: callers test a terminus
    # position for equality against start/end, and an inverted span would answer
    # them with a silent false positive.
    spans = [s for s in spans if s["start"] <= s["end"]]
    return sorted(spans, key=lambda s: (s["start"], s["end"]))


def trimPrecursorSequences(
    mol, sequences: dict, chains=None, tol: float = 2.0, return_offsets: bool = False
) -> "dict | tuple[dict, dict[str, int]]":
    """Trim reference sequences down to the span a structure actually covers.

    A UniProt precursor typically starts before, and can end after, the construct
    in the structure: a signal peptide, an activation peptide, a purification tag.
    Those residues are absent by design, not unresolved, so they must not be
    presented as gaps to model. This aligns each reference against the observed
    sequence and drops the leading and trailing reference residues that align to
    nothing, leaving internal gaps and the residue numbering untouched.

    Only apply this to references that came from UniProt. On an RCSB entity
    sequence a terminal overhang *is* missing density (a disordered terminus) and
    dropping it would silently discard a real, modellable gap -- hence ``chains``.

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The structure the sequences are references for.
    sequences : dict
        ``{chain: full_sequence}``. Not modified; a trimmed copy is returned.
    chains : list of str, optional
        The chains to trim. Defaults to every chain in ``sequences``; pass the
        UniProt-derived chains explicitly when the mapping mixes sources.
    tol : float
        Maximum C-N distance (Angstrom) still counted as a peptide bond when
        locating the structure's backbone breaks, which is what tells the aligner
        where residues can legitimately be missing.
    return_offsets : bool
        Also return ``{chain: n_leading_trimmed}``. Reference position ``p``
        (1-based) in a trimmed sequence is UniProt position ``p + offset``,
        which is how a caller maps a terminus back to precursor numbering.

    Returns
    -------
    trimmed : dict
        ``{chain: sequence}``, with untrimmed chains carried through unchanged.
    offsets : dict
        Only when ``return_offsets`` is True: the number of leading reference
        residues dropped per trimmed chain (0 for chains carried through).

    Examples
    --------
    >>> from moleculekit.uniprot import trimPrecursorSequences, uniprotSequence
    >>> seqs = trimPrecursorSequences(mol, {"A": uniprotSequence("P00760")})  # doctest: +SKIP
    >>> seqs["A"][:4]      # the mature chain, signal + activation peptide gone
    'IVGG'
    """
    from moleculekit.tools.modelling import (
        _align_full_to_observed,
        _observed_backbone_breaks,
    )

    obsseq, obsidx = mol.getSequence(
        dict_key="chain", return_idx=True, sel="protein", _logger=False
    )
    trimmed = dict(sequences)
    offsets = {c: 0 for c in sequences}
    targets = list(sequences) if chains is None else list(chains)
    for chain in targets:
        if chain not in trimmed or not obsseq.get(chain):
            continue
        full = trimmed[chain]
        aln_full, aln_obs = _align_full_to_observed(
            full,
            obsseq[chain],
            breaks=_observed_backbone_breaks(mol, obsidx[chain], tol=tol),
        )
        # Leading / trailing columns where the observed side has nothing: the
        # reference residues in them are the overhang.
        lead = len(aln_obs) - len(aln_obs.lstrip("-"))
        trail = len(aln_obs) - len(aln_obs.rstrip("-"))
        lo = sum(1 for c in aln_full[:lead] if c != "-")
        hi = len(full) - sum(1 for c in aln_full[len(aln_full) - trail :] if c != "-")
        if lo or hi < len(full):
            logger.info(
                f"Chain {chain}: trimmed {lo} N-terminal and {len(full) - hi} "
                f"C-terminal reference residue(s) not present in the structure "
                f"('{full[:lo]}' / '{full[hi:]}')"
            )
        offsets[chain] = lo
        trimmed[chain] = full[lo:hi]
    if return_offsets:
        return trimmed, offsets
    return trimmed
