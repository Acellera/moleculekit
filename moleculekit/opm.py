import numpy as np
from moleculekit.molecule import Molecule
from tqdm import tqdm
import tempfile
import logging
import shutil
import json
import os

logger = logging.getLogger(__name__)


def _filter_opm_pdb(lines, keep_dum=False):
    good_starts = ("ATOM", "HETATM", "MODEL", "ENDMDL")
    newlines = []
    for line in lines:
        if not keep_dum and line[17:20] == "DUM":
            continue
        if any([line.startswith(st) for st in good_starts]):
            # Reduce issues by dropping everything after betas
            newlines.append(line[:67] + "\n")
    return newlines


def generate_opm_sequences(opm_pdbs: list, outjson: str):
    """Extract protein and nucleic sequences from OPM PDB files into an xz-compressed JSON file.

    Each input PDB is filtered of its DUM placeholder atoms, loaded as a
    Molecule, and split into protein and nucleic components. The per-chain
    sequences (dropping chains shorter than 5 residues or consisting only of
    unknown ``X`` residues) are collected and written to an xz-compressed JSON
    file keyed by the PDB file basename. This is used to build the searchable
    database consumed by :func:`align_to_opm`.

    Parameters
    ----------
    opm_pdbs : list of str
        Paths to the OPM PDB files to process.
    outjson : str
        Path to the xz-compressed JSON file that the collected sequences are written to.
    """
    sequences = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        outf = os.path.join(tmpdir, "new.pdb")
        for ff in tqdm(opm_pdbs):
            logger.info(f"Processing {ff}")
            name = os.path.splitext(os.path.basename(ff))[0]
            with open(ff, "r") as f, open(outf, "w") as fout:
                newlines = _filter_opm_pdb(f, keep_dum=False)
                fout.writelines(newlines)

            try:
                mol = Molecule(outf, validateElements=False)
                if mol.numAtoms == 0:
                    continue
                molp = mol.copy()
                molp.filter("protein", _logger=False)
                moln = mol.copy()
                moln.filter("nucleic", _logger=False)
                if molp.numAtoms == 0 and moln.numAtoms == 0:
                    logger.warning("No protein or nucleic found")
                    continue
                sequences[name] = {}
                if molp.numAtoms:
                    seq = molp.getSequence()
                    for k in list(seq.keys()):
                        if len(seq[k]) < 5 or all([ss == "X" for ss in seq[k]]):
                            del seq[k]
                    if len(seq):
                        sequences[name]["protein"] = seq
                if moln.numAtoms:
                    seq = moln.getSequence()
                    for k in list(seq.keys()):
                        if len(seq[k]) < 5 or all([ss == "X" for ss in seq[k]]):
                            del seq[k]
                    if len(seq):
                        sequences[name]["nucleic"] = seq
                if len(sequences[name]) == 0:
                    del sequences[name]
            except Exception as e:
                if name in sequences:
                    del sequences[name]
                logger.warning(f"Failed on file {ff} with error {e}")
                continue

    import lzma

    with lzma.open(outjson, "wt") as f:
        json.dump(sequences, f, indent=4)


def blast_search_opm(query: str, sequences: dict) -> list:
    """Search a query sequence against OPM protein sequences using BLAST+.

    Builds a temporary BLAST protein database from the protein sequences of the
    OPM database and runs ``blastp`` for the query sequence. This requires the
    ``makeblastdb`` and ``blastp`` executables (from BLAST+) to be available on
    the system PATH.

    Parameters
    ----------
    query : str
        The query protein sequence to search for.
    sequences : dict
        The OPM sequences database, as produced by
        :func:`generate_opm_sequences`, keyed by PDB id with a ``"protein"``
        entry mapping chain ids to sequences.

    Returns
    -------
    hits : list
        The BLAST search hits, as parsed from the ``blastp`` JSON output.
    """
    import json
    from subprocess import call
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        fastaf = os.path.join(tmpdir, "opm.faa")
        with open(fastaf, "w") as f:
            for pdbid in sequences:
                for chain in sequences[pdbid]["protein"]:
                    f.write(f">pdb|{pdbid.upper()}|{chain}\n")
                    f.write(sequences[pdbid]["protein"][chain].replace("?", "X") + "\n")

        call(["makeblastdb", "-dbtype", "prot", "-in", fastaf])

        queryf = os.path.join(tmpdir, "query.pro")
        with open(queryf, "w") as f:
            f.write(f">query\n{query}")

        outf = os.path.join(tmpdir, "output.json")
        call(["blastp", "-out", outf, "-outfmt", "15", "-query", queryf, "-db", fastaf])

        with open(outf, "r") as f:
            results = json.load(f)

    return results["BlastOutput2"][0]["report"]["results"]["search"]["hits"]


def _opm_chain_sequences(sequences: dict, pdbids=None):
    """``(pdbid, chain, sequence)`` for every protein chain, X for unknowns."""
    for pdbid in sequences if pdbids is None else pdbids:
        for chain, seq in (sequences[pdbid]["protein"] or {}).items():
            seq = seq.replace("?", "X")
            if seq:
                yield pdbid, chain, seq


def _rcsb_opm_candidates(query: str, sequences: dict, rows: int = 25) -> list:
    """OPM ids whose sequence RCSB's hosted search matches to ``query``."""
    from moleculekit.rcsb import rcsbSequenceSearch

    hits = rcsbSequenceSearch(query, identity_cutoff=0.3, rows=rows)
    seen, out = set(), []
    for hit in hits:
        pdbid = hit["polymer_entity_id"].split("_")[0].lower()
        if pdbid in sequences and pdbid not in seen:
            seen.add(pdbid)
            out.append(pdbid)
    return out


def pairwise_search_opm(query: str, sequences: dict, maxhits: int = 3) -> list:
    """Search a query sequence against OPM protein sequences with Biopython.

    Scores the query against every chain in ``sequences`` with a local
    (Smith-Waterman) alignment, then returns the best ``maxhits`` in the shape
    :func:`align_to_opm` consumes: one entry per hit carrying a ``description``
    title and its ``hsps`` records. Offsets are 1-based, start inclusive.

    Parameters
    ----------
    query : str
        The one-letter query protein sequence.
    sequences : dict
        The OPM sequences database, keyed by PDB id with a ``"protein"`` entry
        mapping chain ids to sequences.
    maxhits : int
        How many of the best-scoring chains to align and return.

    Returns
    -------
    hits : list of dict
        Best-scoring first.
    """
    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner(scoring="blastp")
    aligner.mode = "local"

    scored = sorted(
        (
            (aligner.score(query, seq), pdbid, chain, seq)
            for pdbid, chain, seq in _opm_chain_sequences(sequences)
        ),
        key=lambda row: -row[0],
    )

    hits = []
    for score, pdbid, chain, seq in scored[:maxhits]:
        hsps = _local_hsps(aligner, query, seq)
        if hsps:
            hits.append(
                {
                    "description": [{"title": f"pdb|{pdbid.upper()}|{chain}"}],
                    "hsps": hsps,
                }
            )
    return hits


def _local_hsps(aligner, query, target, min_length: int = 20, max_hsps: int = 10):
    """Non-overlapping local alignments of ``query`` onto ``target``.

    The caller runs one structural alignment per pair, so a query holding two
    copies of the reference chain yields a candidate orientation per copy.
    ``PairwiseAligner`` returns a single alignment, so the query is searched
    again either side of each match until no segment is long enough to hold
    one.

    Offsets are 1-based, start inclusive.
    """
    hsps: list = []
    pending = [(0, len(query))]
    while pending and len(hsps) < max_hsps:
        scored = []
        for start, end in pending:
            if end - start < min_length:
                continue
            alignments = aligner.align(query[start:end], target)
            if len(alignments):
                scored.append((alignments[0].score, start, end, alignments[0]))
        if not scored:
            break
        score, start, end, alignment = max(scored, key=lambda row: row[0])
        qblocks, tblocks = alignment.aligned
        if not len(qblocks):
            break
        qfrom, qto = int(qblocks[0][0]) + start, int(qblocks[-1][1]) + start
        tfrom, tto = int(tblocks[0][0]), int(tblocks[-1][1])
        pending.remove((start, end))
        pending.extend(
            seg for seg in ((start, qfrom), (qto, end)) if seg[1] - seg[0] >= min_length
        )
        if qto - qfrom < min_length:
            # A match shorter than the window is the tail of the search
            # finding noise rather than another copy.
            continue
        hsps.append(
            {
                "query_from": qfrom + 1,
                "query_to": qto,
                "hit_from": tfrom + 1,
                "hit_to": tto,
                "align_len": qto - qfrom,
                "bit_score": float(score),
                "evalue": float("nan"),
            }
        )
    return hsps


def get_opm_pdb(
    pdbid: str,
    keep: bool = False,
    keepaltloc: str = "A",
    validateElements: bool = False,
):
    """Download a membrane system from the OPM.

    Parameters
    ----------
    pdbid : str
        The 4-letter PDB code
    keep: bool
        If False, removes the DUM atoms. If True, it keeps them.
    keepaltloc : str
        Which altloc to keep if there are any
    validateElements : bool
        Set to True to validate the elements read. Usually this will fail on OPM due to weird atom names

    Returns
    -------
    mol: Molecule
        The oriented molecule

    thickness: float or None
        The bilayer thickness (both layers)

    Examples
    --------
    >>> mol, thickness = get_opm_pdb("1z98")
    >>> mol.numAtoms
    7902
    >>> thickness
    28.2
    >>> _, thickness = get_opm_pdb('4u15')
    >>> thickness is None
    True

    """
    import requests
    import re
    from moleculekit.molecule import Molecule

    try:
        resp = requests.get(
            f"https://biomembhub.org/shared/opm-assets/pdb/{pdbid.lower()}.pdb"
        )
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch OPM with PDB ID {pdbid} with error {e}")

    lines = resp.text.splitlines()
    # Assuming the half-thickness is the last word in the matched line
    # REMARK      1/2 of bilayer thickness:   14.1
    pattern = re.compile("^REMARK.+thickness")
    thickness = None
    for line in lines:
        if re.match(pattern, line):
            thickness = 2.0 * float(line.split()[-1])
            break

    newlines = _filter_opm_pdb(lines, keep_dum=keep)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpf = os.path.join(tmpdir, "opm.pdb")
        with open(tmpf, "w") as f:
            f.writelines(newlines)

        mol = Molecule(tmpf, keepaltloc=keepaltloc, validateElements=validateElements)

    return mol, thickness


def align_to_opm(
    mol: "Molecule",
    molsel: str | np.ndarray = "all",
    maxalignments: int = 3,
    opmid: str | None = None,
    macrotype: str = "protein",
) -> list:
    """Align a Molecule to proteins/nucleics in the OPM database by sequence search

    The matching entry is found with RCSB's hosted sequence search, and the
    residue ranges it is aligned over with a local Biopython alignment. Passing
    ``opmid`` skips the search entirely.

    Parameters
    ----------
    mol : Molecule
        The query molecule. The alignments will be done on the first frame only.
    molsel : str or np.ndarray
        The atom selection for the query molecule to use. Can be an atom selection
        string, a boolean mask, or an integer index array.
    maxalignments : int
        The maximum number of aligned structures to return
    opmid : str
        An OPM ID to align onto. Passing one skips the search: the database is
        narrowed to that entry and only the residue ranges are computed.
    macrotype : str
        If to align on "protein" or "nucleic"

    Returns
    -------
    results : list of dictionaries
        Returns a number of alignments (maximum `maxalignments`). For each alignment
        it might contain a number of HSPs (high-scoring pairs) which correspond to different
        sequence alignments of the query on the same hit protein.

    """
    from moleculekit import __share_dir
    from moleculekit.align import molTMalign

    import lzma

    with lzma.open(os.path.join(__share_dir, "opm_sequences.json.xz"), "rt") as f:
        sequences = json.load(f)

    if opmid is not None:
        if opmid.lower() not in sequences:
            raise RuntimeError(f"Could not find {opmid} in OPM database")
        # Throw away all other sequences
        sequences = {opmid.lower(): sequences[opmid.lower()]}

    seqmol, molidx = mol.getSequence(
        dict_key=None, return_idx=True, sel=molsel, _logger=False
    )
    seqmol = seqmol[macrotype]
    molidx = molidx[macrotype]

    if opmid is None:
        # The hosted search picks the candidates out of 15k chains in one call;
        # scoring them all locally takes tens of seconds, so that is the
        # fallback rather than the default.
        try:
            candidates = _rcsb_opm_candidates(seqmol, sequences)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"RCSB sequence search failed ({e}), scoring OPM locally")
            candidates = None
        if candidates is not None:
            if not candidates:
                raise RuntimeError("No OPM entry matched the query sequence")
            sequences = {p: sequences[p] for p in candidates}

    res = pairwise_search_opm(seqmol, sequences, maxhits=maxalignments)

    all_aligned_structs = []
    for i in range(min(maxalignments, len(res))):
        rr = res[i]
        pdbid = rr["description"][0]["title"].split("|")[1]
        logger.info(
            f"Sequence match with OPM {pdbid}. {len(rr['hsps'])} high-scoring pairs (HSPs)"
        )
        ref, thickness = get_opm_pdb(pdbid, validateElements=False)

        seqref, refidx = ref.getSequence(dict_key=None, return_idx=True, _logger=False)
        seqref = seqref[macrotype]
        refidx = refidx[macrotype]

        alignedstructs = []
        for j, hsp in enumerate(rr["hsps"]):  # Iterate highest-scoring-pairs
            molidx_hsp = np.hstack(molidx[hsp["query_from"] : hsp["query_to"]])
            refidx_hsp = np.hstack(refidx[hsp["hit_from"] : hsp["hit_to"]])
            molidx_sel = f"index {' '.join(map(str, molidx_hsp))} and name CA"
            refidx_sel = f"index {' '.join(map(str, refidx_hsp))} and name CA"
            try:
                t0, rmsd, nali, alncoo, _ = molTMalign(mol, ref, molidx_sel, refidx_sel)
            except Exception:
                logger.warning(f"Failed to generate alignment for HSP {j}")
                continue
            molc = mol.copy()

            molc.coords[:] = alncoo
            alignedstructs.append(
                {"aligned_mol": molc, "TM-Score": t0[0], "Common RMSD": rmsd[0]}
            )
            logger.info(
                f"   HSP {j} length {hsp['align_len']}, e-value {hsp['evalue']}, TM-score {t0[0]:.2f}, RMSD {rmsd[0]:.2f}, res_aligned {nali[0]}"
            )

        all_aligned_structs.append(
            {"hsps": alignedstructs, "pdbid": pdbid, "thickness": thickness}
        )
    return all_aligned_structs
