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


def trimPrecursorSequences(mol, sequences: dict, chains=None, tol: float = 2.0) -> dict:
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

    Returns
    -------
    trimmed : dict
        ``{chain: sequence}``, with untrimmed chains carried through unchanged.

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
        trimmed[chain] = full[lo:hi]
    return trimmed
