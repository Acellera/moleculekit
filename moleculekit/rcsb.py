# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import urllib.request
import logging

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
