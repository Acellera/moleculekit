import urllib.request
import logging
logger = logging.getLogger(__name__)


def _getRCSBtext(url):
    connected = False
    while not connected:
        try:
            response = urllib.request.urlopen(url)
            text = response.read()
        except Exception as coer:
            import time
            logger.warning('Failed to connect to URL {} with error {}. Sleeping 5s and retrying.'.format(url, coer))
            time.sleep(5)
            continue
        connected = True

    return text


def rcsbFindMutatedResidues(pdbid):
    """
    Examples
    --------
    >>> rcsbFindMutatedResidues('3onq')
    {'MSE': 'MET'}
    """
    try:
        from bs4 import BeautifulSoup
        import lxml
    except ImportError:
        raise ImportError('You need to install the \'beautifulsoup4\' and \'lxml\' packages to use this function.')
    tomutate = {}

    url = 'http://www.rcsb.org/pdb/explore.do?structureId={}'.format(pdbid)
    text = _getRCSBtext(url)
    soup = BeautifulSoup(text, 'lxml')
    table = soup.find(id='ModifiedResidueTable')

    if table:
        trs = table.find_all('tr')

        for tr in trs:
            td = tr.find_all('td')
            if td:
                mutname = td[0].find_all('a')[0].text.strip()
                orgname = td[5].text.strip()
                tomutate[mutname] = orgname
    return tomutate


def rcsbFindLigands(pdbid):
    """
    Examples
    --------
    >>> rcsbFindLigands('3onq')
    ['SO4', 'GOL']
    """
    try:
        from bs4 import BeautifulSoup
        import lxml
    except ImportError:
        raise ImportError('You need to install the \'beautifulsoup4\' and \'lxml\' packages to use this function.')
    ligands = []

    url = 'http://www.rcsb.org/pdb/explore.do?structureId={}'.format(pdbid)
    text = _getRCSBtext(url)
    soup = BeautifulSoup(text, 'lxml')
    table = soup.find(id='LigandsTable')
    if table:
        trs = table.find_all('tr')

        for tr in trs:
            td = tr.find_all('td')
            if td:
                name = td[0].find_all('a')[0].text.strip()
                ligands.append(name)
    return ligands