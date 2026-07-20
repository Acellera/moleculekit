# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import urllib.request
import urllib.error
import json
import logging

from moleculekit.util import _get_pdb_entity_sequences

logger = logging.getLogger(__name__)


def _getRCSBtext(url, attempts=3):
    connected = False
    for _ in range(attempts):
        try:
            response = urllib.request.urlopen(url)
            text = response.read()
        except Exception as coer:
            import time

            logger.warning(
                f"Failed to connect to URL {url} with error {coer}. Sleeping 5s and retrying."
            )
            time.sleep(5)
            continue
        connected = True

    if not connected:
        raise RuntimeError(f"Failed to connect to URL {url}")

    return text


def fetchResidueCIF(resname: str, outdir: str, overwrite: bool = False) -> str:
    """Download a residue's reference structure from the RCSB Chemical Component
    Dictionary and store it as ``<resname>.cif`` in ``outdir``.

    Used at packaging time to populate moleculekit's ``share/residue_cifs`` with
    the modified-residue templates that ``systemPrepare`` injects into PDB2PQR so
    those residues can be protonated. NOT called at runtime - runtime reads the
    already-shipped cifs. No-op if the file exists unless ``overwrite`` is set.

    Parameters
    ----------
    resname : str
        The PDB chemical-component code, e.g. ``"HYP"``.
    outdir : str
        Directory to write ``<resname>.cif`` into.
    overwrite : bool
        Re-download even if the file already exists.

    Returns
    -------
    path : str
        Path to the written (or pre-existing) cif file.
    """
    import os

    outpath = os.path.join(outdir, f"{resname}.cif")
    if os.path.isfile(outpath) and not overwrite:
        return outpath
    url = f"https://files.rcsb.org/ligands/download/{resname}.cif"
    text = _getRCSBtext(url)
    if isinstance(text, bytes):
        text = text.decode()
    os.makedirs(outdir, exist_ok=True)
    with open(outpath, "w") as fh:
        fh.write(text)
    return outpath


def rcsbFindMutatedResidues(pdbid: str) -> dict:
    """Find the modified/mutated residues of a PDB entry.

    Scrapes the RCSB PDB entry page for its table of modified residues and maps
    each non-standard residue name to the standard residue it derives from.

    Parameters
    ----------
    pdbid : str
        The 4-letter PDB code to look up.

    Returns
    -------
    tomutate : dict
        A mapping from each modified residue name to its parent standard residue
        name (e.g. ``{'MSE': 'MET'}``). Empty if no modified residues are found.

    Examples
    --------
    >>> rcsbFindMutatedResidues('3onq')
    {'MSE': 'MET'}
    """
    try:
        from bs4 import BeautifulSoup
        import lxml
    except ImportError:
        raise ImportError(
            "You need to install the 'beautifulsoup4' and 'lxml' packages to use this function."
        )
    tomutate = {}

    url = f"http://www.rcsb.org/pdb/explore.do?structureId={pdbid}"
    text = _getRCSBtext(url)
    soup = BeautifulSoup(text, "lxml")
    table = soup.find(id="ModifiedResidueTable")

    if table:
        trs = table.find_all("tr")

        for tr in trs:
            td = tr.find_all("td")
            if td:
                mutname = td[0].find_all("a")[0].text.strip()
                orgname = td[5].text.strip()
                tomutate[mutname] = orgname
    return tomutate


def rcsbFindLigands(pdbid: str) -> list:
    """Find the ligands present in a PDB entry.

    Scrapes the RCSB PDB entry page for its table of ligands and returns their
    residue names.

    Parameters
    ----------
    pdbid : str
        The 4-letter PDB code to look up.

    Returns
    -------
    ligands : list of str
        The residue names of the ligands found in the entry (e.g.
        ``['SO4', 'GOL']``). Empty if no ligands are found.

    Examples
    --------
    >>> rcsbFindLigands('3onq')
    ['SO4', 'GOL']
    """
    try:
        from bs4 import BeautifulSoup
        import lxml
    except ImportError:
        raise ImportError(
            "You need to install the 'beautifulsoup4' and 'lxml' packages to use this function."
        )
    ligands = []

    url = f"http://www.rcsb.org/pdb/explore.do?structureId={pdbid}"
    text = _getRCSBtext(url)
    soup = BeautifulSoup(text, "lxml")
    table = soup.find(id="LigandsTable")
    if table:
        trs = table.find_all("tr")

        for tr in trs:
            td = tr.find_all("td")
            if td:
                name = td[0].find_all("a")[0].text.strip()
                ligands.append(name)
    return ligands


def _getRCSBjson(url, attempts=3):
    import time

    last_err = None
    for _ in range(attempts):
        try:
            response = urllib.request.urlopen(url)
            return json.loads(response.read())
        except urllib.error.HTTPError as err:
            # 404 etc. — the code is wrong, do not retry
            raise RuntimeError(f"RCSB request failed for {url}: {err}") from err
        except Exception as err:
            last_err = err
            logger.warning(
                f"Failed to connect to URL {url} with error {err}. Sleeping 5s and retrying."
            )
            time.sleep(5)
    raise RuntimeError(f"Failed to connect to URL {url}: {last_err}")


def rcsbFetchLigandInfo(comp_id: str) -> dict:
    """Fetch the full RCSB Chemical Component Dictionary record for a ligand.

    Queries the RCSB data API for a 3-letter chemical component (CCD) code and
    returns the complete record, including identifiers, formula, weight and all
    descriptor variants (InChI plus SMILES from RCSB, CACTVS, OpenEye and ACDLabs).

    Parameters
    ----------
    comp_id : str
        The chemical component (CCD) 3-letter code, e.g. ``"BEN"``. Case-insensitive.

    Returns
    -------
    info : dict
        The parsed JSON record. The curated descriptors live under
        ``info["rcsb_chem_comp_descriptor"]`` (``SMILES``, ``SMILES_stereo``,
        ``InChI``, ``InChIKey``); per-program variants live under
        ``info["pdbx_chem_comp_descriptor"]``.

    Examples
    --------
    >>> info = rcsbFetchLigandInfo('BEN')
    >>> info['rcsb_chem_comp_descriptor']['comp_id']
    'BEN'
    """
    comp_id = comp_id.strip().upper()
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}"
    return _getRCSBjson(url)


def rcsbFetchLigandSmiles(
    comp_id: str, stereo: bool = True, program: str = "OpenEye"
) -> str:
    """Fetch a SMILES string for a ligand by its RCSB CCD code.

    Thin wrapper over :func:`rcsbFetchLigandInfo`. RCSB stores SMILES computed by
    several toolkits (OpenEye, CACTVS, ACDLabs). By default this returns the
    OpenEye descriptor, which RCSB also curates into its top-level
    ``rcsb_chem_comp_descriptor`` block. Pass ``program`` to pick a different
    toolkit; for full control read ``pdbx_chem_comp_descriptor`` off
    :func:`rcsbFetchLigandInfo` directly.

    Parameters
    ----------
    comp_id : str
        The chemical component (CCD) 3-letter code, e.g. ``"BEN"``. Case-insensitive.
    stereo : bool
        If True (default) return the isomeric SMILES (stereochemistry included);
        if False return the plain SMILES. RCSB labels the isomeric variant
        ``SMILES_CANONICAL``. Falls back to the other variant when the preferred
        one is absent for the chosen program.
    program : str
        Which toolkit's descriptor to return. ``"OpenEye"`` (default) uses RCSB's
        curated descriptor. Other typical values are ``"CACTVS"`` and
        ``"ACDLabs"``. Matched case-insensitively as a substring of the program
        name reported by RCSB; raises if the component has no SMILES from a
        matching program.

    Returns
    -------
    smiles : str
        The SMILES string.

    Examples
    --------
    >>> rcsbFetchLigandSmiles('BEN', stereo=False)
    '[H]N=C(c1ccccc1)N'
    >>> rcsbFetchLigandSmiles('BEN', program='CACTVS')
    'NC(=N)c1ccccc1'
    """
    info = rcsbFetchLigandInfo(comp_id)
    code = comp_id.strip().upper()
    want = program.strip().lower()

    # OpenEye is the default, and is exactly what RCSB curates into the top-level
    # ``rcsb_chem_comp_descriptor`` block (always present) — use it directly.
    if want in ("openeye", "openeye oetoolkits", "oe"):
        desc = info.get("rcsb_chem_comp_descriptor", {})
        primary, secondary = (
            ("SMILES_stereo", "SMILES") if stereo else ("SMILES", "SMILES_stereo")
        )
        smiles = desc.get(primary) or desc.get(secondary)
        if smiles:
            return smiles
        # else fall through to the per-program rows below

    # Per-program descriptors (CACTVS, ACDLabs, or an OpenEye fallback).
    rows = [
        r
        for r in info.get("pdbx_chem_comp_descriptor", [])
        if "SMILES" in (r.get("type") or "")
    ]
    available = sorted({r.get("program") for r in rows if r.get("program")})
    matches = [r for r in rows if want and want in (r.get("program") or "").lower()]
    if not matches:
        raise RuntimeError(
            f"RCSB has no SMILES for component '{code}' from program '{program}'. "
            f"Available programs: {available}"
        )
    # ``SMILES_CANONICAL`` is the isomeric (stereo-bearing) variant.
    primary, secondary = (
        ("SMILES_CANONICAL", "SMILES") if stereo else ("SMILES", "SMILES_CANONICAL")
    )
    by_type = {r.get("type"): r.get("descriptor") for r in matches}
    smiles = by_type.get(primary) or by_type.get(secondary)
    if not smiles:
        raise RuntimeError(
            f"RCSB returned no SMILES descriptor for component '{code}' from program '{program}'"
        )
    return smiles


def rcsbSequenceSearch(
    sequence: str, identity_cutoff: float = 0.9, rows: int = 10
) -> list:
    """Search the RCSB hosted sequence-similarity service for a protein sequence.

    Parameters
    ----------
    sequence : str
        The one-letter protein query sequence.
    identity_cutoff : float
        Minimum sequence identity (0-1) for a hit to be returned.
    rows : int
        Maximum number of hits to return.

    Returns
    -------
    hits : list of dict
        ``{"polymer_entity_id": str, "identity": float, "score": float}``,
        ordered best-first.
    """
    import urllib.parse

    query = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 1,
                "identity_cutoff": identity_cutoff,
                "sequence_type": "protein",
                "value": sequence,
            },
        },
        "request_options": {
            "scoring_strategy": "sequence",
            "paginate": {"start": 0, "rows": rows},
        },
        "return_type": "polymer_entity",
    }
    url = "https://search.rcsb.org/rcsbsearch/v2/query?json=" + urllib.parse.quote(
        json.dumps(query)
    )
    with urllib.request.urlopen(url, timeout=45) as resp:
        data = json.loads(resp.read())

    hits = []
    for r in data.get("result_set", []):
        identity = None
        try:
            identity = r["services"][0]["nodes"][0]["match_context"][0][
                "sequence_identity"
            ]
        except (KeyError, IndexError):
            identity = None
        hits.append(
            {
                "polymer_entity_id": r["identifier"],
                "identity": identity,
                "score": r.get("score"),
            }
        )
    return hits


def _entity_sequences_for_pdbid(pdbid):
    """Map each auth chain of a PDB entry to its full deposited (canonical)
    sequence via the RCSB GraphQL API. Returns ``{chain: sequence}``."""
    import urllib.parse

    q = (
        '{entries(entry_ids:["%s"]){polymer_entities{'
        "entity_poly{pdbx_seq_one_letter_code_can} "
        "rcsb_polymer_entity_container_identifiers{auth_asym_ids}}}}" % pdbid.upper()
    )
    url = "https://data.rcsb.org/graphql?query=" + urllib.parse.quote(q)
    with urllib.request.urlopen(url, timeout=45) as resp:
        data = json.loads(resp.read())
    out = {}
    entries = data["data"]["entries"] or []
    for ent in (entries[0]["polymer_entities"] if entries else []) or []:
        seq = ent["entity_poly"]["pdbx_seq_one_letter_code_can"]
        for ch in ent["rcsb_polymer_entity_container_identifiers"]["auth_asym_ids"]:
            out[ch] = seq
    return out


def resolveFullSequences(mol, pdbid=None):
    """Resolve the full target sequence of each protein chain in ``mol``.

    When ``pdbid`` is given, the exact deposited entity sequence is used
    (``source="pdb_entity"``, ``identity=1.0``). Otherwise each chain's observed
    sequence is run through :func:`rcsbSequenceSearch` and the best hit's full
    entity sequence is used (``source="sequence_search"``).

    Parameters
    ----------
    mol : :class:`Molecule <moleculekit.molecule.Molecule>`
        The (possibly gapped) structure.
    pdbid : str or None
        The 4-letter RCSB PDB id, if known.

    Returns
    -------
    resolved : dict
        ``{chain: {"sequence": str, "source": str, "identity": float}}`` for each
        protein chain for which a full sequence could be found.
    """
    observed = mol.getSequence(dict_key="chain", sel="protein", _logger=False)
    resolved = {}

    entity_seqs = _entity_sequences_for_pdbid(pdbid) if pdbid else {}
    for chain, obs in observed.items():
        if not obs:
            continue
        if chain in entity_seqs:
            resolved[chain] = {
                "sequence": entity_seqs[chain],
                "source": "pdb_entity",
                "identity": 1.0,
            }
            continue
        hits = rcsbSequenceSearch(obs.replace("X", ""))
        if not hits:
            continue
        best = hits[0]
        ent_map = _get_pdb_entity_sequences([best["polymer_entity_id"]])
        full = next(iter(ent_map.values()), None)
        if full is None:
            continue
        # _get_pdb_entity_sequences renders modified residues as "?"; map them to
        # "X" (unknown) so the downstream BLOSUM62 aligner accepts the sequence.
        full = full.replace("?", "X")
        resolved[chain] = {
            "sequence": full,
            "source": "sequence_search",
            "identity": best["identity"],
        }
    return resolved
