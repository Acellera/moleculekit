from moleculekit.molecule import Molecule
from tqdm import tqdm
import tempfile
import logging
import shutil
import json
import os

logger = logging.getLogger(__name__)

blastp = shutil.which("blastp", mode=os.X_OK)


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


def generate_opm_sequences(opm_pdbs, outjson):
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
                    seq = molp.sequence()
                    for k in list(seq.keys()):
                        if len(seq[k]) < 5 or all([ss == "X" for ss in seq[k]]):
                            del seq[k]
                    if len(seq):
                        sequences[name]["protein"] = seq
                if moln.numAtoms:
                    seq = moln.sequence()
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

    with open(outjson, "w") as f:
        json.dump(sequences, f, indent=4)


def blast_search_opm(query, sequences):
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


def get_opm_pdb(pdbid, keep=False, keepaltloc="A", validateElements=False):
    """Download a membrane system from the OPM.

    Parameters
    ----------
    pdb: str
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
            f"https://storage.googleapis.com/opm-assets/pdb/{pdbid.lower()}.pdb"
        )
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


def align_to_opm(mol, molsel="all", maxalignments=3, opmid=None, macrotype="protein"):
    """Align a Molecule to proteins/nucleics in the OPM database by sequence search

    This function requires BLAST+ to be installed. You can find the latest BLAST executables here:
    https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/
    Once you have it installed, export it to your PATH before starting python so that it's able to
    detect the blastp and makeblastdb executables.
    Alternatively install it via `conda install blast -c bioconda`

    Parameters
    ----------
    mol : Molecule
        The query molecule. The alignments will be done on the first frame only.
    molsel : str
        The atom selection for the query molecule to use
    maxalignments : int
        The maximum number of aligned structures to return
    opmid : str
        If an OPM ID is passed the function will skip searching the database
    macrotype : str
        If to align on "protein" or "nucleic"

    Returns
    -------
    results : list of dictionaries
        Returns a number of alignements (maximum `maxalignments`). For each alignment
        it might contain a number of HSPs (high-scoring pairs) which correspond to different
        sequence alignments of the query on the same hit protein.

    """
    from moleculekit import __share_dir
    from moleculekit.align import molTMalign
    import numpy as np

    with open(os.path.join(__share_dir, "opm_sequences.json"), "r") as f:
        sequences = json.load(f)

    if opmid is not None:
        if opmid.lower() not in sequences:
            raise RuntimeError(f"Could not find {opmid} in OPM database")
        # Throw away all other sequences
        sequences = {opmid.lower(): sequences[opmid.lower()]}

    seqmol, molidx = mol.sequence(
        noseg=True, return_idx=True, sel=molsel, _logger=False
    )
    seqmol = seqmol[macrotype]
    molidx = molidx[macrotype]

    res = blast_search_opm(seqmol, sequences)

    all_aligned_structs = []
    for i in range(min(maxalignments, len(res))):
        rr = res[i]
        pdbid = rr["description"][0]["title"].split("|")[1]
        logger.info(
            f"Sequence match with OPM {pdbid}. {len(rr['hsps'])} high-scoring pairs (HSPs)"
        )
        ref, thickness = get_opm_pdb(pdbid, validateElements=False)

        seqref, refidx = mol.sequence(noseg=True, return_idx=True, _logger=False)
        seqref = seqref[macrotype]
        refidx = refidx[macrotype]

        alignedstructs = []
        for j, hsp in enumerate(rr["hsps"]):  # Iterate highest-scoring-pairs
            molidx_hsp = np.hstack(molidx[hsp["query_from"] : hsp["query_to"]])
            refidx_hsp = np.hstack(refidx[hsp["hit_from"] : hsp["hit_to"]])
            molidx_sel = f"index {' '.join(map(str, molidx_hsp))} and name CA"
            refidx_sel = f"index {' '.join(map(str, refidx_hsp))} and name CA"
            t0, rmsd, nali, aln, _ = molTMalign(mol, ref, molidx_sel, refidx_sel)
            molc = mol.copy()
            molc.coords = aln[0].copy()
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
