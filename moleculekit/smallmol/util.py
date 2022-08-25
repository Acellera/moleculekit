# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import os
import numpy as np
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

_highlight_colors = [
    (1.00, 0.50, 0.00),
    (0.00, 0.50, 1.00),
    (0.00, 1.00, 0.50),
    (1.00, 0.00, 0.50),
    (0.50, 0.00, 1.00),
    (0.50, 1.00, 0.00),
    (1.00, 0.00, 0.25),
    (0.00, 0.25, 1.00),
    (0.25, 1.00, 0.00),
]


fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)


def getRCSBLigandByLigname(ligname, returnMol2=False):
    """
    Returns a SmallMol object of a ligand by its three letter lignane. This molecule is retrieve from RCSB and a mol2
    written. It is possible to return also the mol2 filename.

    Parameters
    ----------
    ligname: str
        The three letter ligand name
    returnMol2: bool
        If True, the mol2 filename is returned

    Returns
    -------
    sm: moleculekit.smallmol.smallmol.SmallMol
        The SmallMol object

    mol2filename: str
        The mol2 filename

    Example
    -------
    >>> from moleculekit.molecule import Molecule
    >>> mol = Molecule('4eiy')
    >>> np.unique(mol.get('resname', 'not protein and not water'))
    array(['CLR', 'NA', 'OLA', 'OLB', 'OLC', 'PEG', 'ZMA'], dtype=object)
    >>> sm = getRCSBLigandByLigname('ZMA')  # doctest: +ELLIPSIS
    SmallMol module...
    >>> sm.numAtoms
    40
    >>> sm, mol2filename = getRCSBLigandByLigname('ZMA', returnMol2=True)
    >>> mol2filename  # doctest: +ELLIPSIS
    '/tmp/tmp....mol2'

    """
    from moleculekit.util import string_to_tempfile
    from moleculekit.smallmol.smallmol import SmallMol
    from moleculekit.rcsb import _getRCSBtext
    from moleculekit.tools.obabel_tools import openbabelConvert

    url = f"https://files.rcsb.org/ligands/view/{ligname}_ideal.sdf"
    sdf_text = _getRCSBtext(url).decode("ascii")
    tempfile = string_to_tempfile(sdf_text, "sdf")
    mol2 = openbabelConvert(tempfile, "sdf", "mol2")

    sm = SmallMol(mol2)
    if returnMol2:
        return sm, mol2

    return sm


def getChemblLigandByDrugName(drugname, returnSmile=False):
    """
    Returns a SmallMol object of a ligand by its drug name. This molecule is retrieve from Chembl. It is possible to
    return also the smile of the ligand.

    Parameters
    ----------
    drugname: str
        The drug name
    returnSmile: bool
        If True, the smile is returned

    Returns
    -------
    sm: moleculekit.smallmol.smallmol.SmallMol
        The SmallMol object

    smile: str
        The smile

    Example
    -------
    >>> sm = getChemblLigandByDrugName('paracetamol')  # doctest: +SKIP
    >>> sm.numAtoms  # doctest: +SKIP
    20
    >>> sm, smile = getChemblLigandByDrugName('paracetamol', returnSmile=True)  # doctest: +SKIP
    >>> smile  # doctest: +SKIP
    'CC(=O)Nc1ccc(O)cc1'
    """
    from moleculekit.smallmol.smallmol import SmallMol

    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        raise ImportError(
            "You need to install the chembl_webresource package to use this function. Try using `conda install "
            "-c chembl chembl_webresource_client`."
        )
    drug = new_client.drug
    results = drug.filter(synonyms__icontains=drugname)

    chembl_id = None

    if len(results) == 0:
        return None

    found = False
    for drug_chembl in results:
        for name in drug_chembl["synonyms"]:
            matched = [True for na in name.split() if na.lower() == drugname.lower()]
            if sum(matched) != 0:
                found = True
                chembl_id = drug_chembl["molecule_chembl_id"]
                break
            if found:
                break
    molecule = new_client.molecule
    molecule_chembl = molecule.get(chembl_id)
    smi = molecule_chembl["molecule_structures"]["canonical_smiles"]
    sm = SmallMol(smi)
    if returnSmile:
        return sm, smi
    return sm


def getChemblSimilarLigandsBySmile(smi, threshold=85, returnSmiles=False):
    """
    Returns a SmallMolLib object of the ligands having a similarity with a smile of at least the specified
    threshold.. This molecules are retrieve from Chembl. It is possible to return also the list smiles.

    Parameters
    ----------
    smi: str
        The smile
    threshold: int
        The threshold value to apply for the similarity search
    returnSmiles: bool
        If True, the list smiles is returned

    Returns
    -------
    sm: moleculekit.smallmol.smallmol.SmallMol
        The SmallMol object

    smiles: str
        The list of smiles

    Example
    -------
    >>> _, smile = getChemblLigandByDrugName('ibuprofen', returnSmile=True)  # doctest: +SKIP
    >>> lib = getChemblSimilarLigandsBySmile(smile)  # doctest: +SKIP
    >>> lib.numMols  # doctest: +SKIP
    4
    >>> lib, smiles = getChemblSimilarLigandsBySmile(smile, returnSmiles=True)  # doctest: +SKIP
    >>> len(smiles)  # doctest: +SKIP
    4
    """
    from moleculekit.smallmol.smallmol import SmallMol
    from moleculekit.smallmol.smallmollib import SmallMolLib

    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        raise ImportError(
            "You need to install the chembl_webresource package to use this function. Try using `conda install "
            "-c chembl chembl_webresource_client`."
        )

    smi_list = []

    similarity = new_client.similarity
    results = similarity.filter(smiles=smi, similarity=threshold).only(
        ["molecule_structures"]
    )
    results = results.all()
    for r in range(len(results)):
        tmp_smi = results[r]["molecule_structures"]["canonical_smiles"]
        fragments = tmp_smi.split(".")
        fragments_len = [len(fr) for fr in fragments]
        fragment = fragments[fragments_len.index(max(fragments_len))]

        if fragment not in smi_list:
            smi_list.append(fragment)

    lib = SmallMolLib()
    for smi in smi_list:
        lib.appendSmallMol(SmallMol(smi))

    if returnSmiles:
        return lib, smi_list

    return lib


def convertToString(arr):

    if isinstance(arr, list):
        arr_str = " ".join([str(i) for i in arr])
    elif isinstance(arr, tuple):
        arr_str = " ".join([str(i) for i in arr])
    else:
        arr_str = " ".join([str(i) for i in arr[0]])

    return arr_str


def _depictMol(
    mol,
    filename=None,
    ipython=False,
    atomlabels=None,
    highlightAtoms=None,
    resolution=(400, 200),
):
    """
    Returns the image or the ipython rendering.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
        The rdkit molecule to depict
    filename: str
        The filename of the image
    ipython: bool
        If True, the SVG rendering for jupiter-nootebook are returned
    atomlabels: list
        List of the label to use for each atom
    highlightAtoms: list
        List of atom index to highlight. Can be also list of list for different selection-colors

    Returns
    -------
    svg: SVG
        If ipython set as True, the SVG rendering is returned

    """
    from os.path import splitext
    from rdkit.Chem import Kekulize
    from rdkit.Chem.Draw import rdMolDraw2D
    from IPython.display import SVG

    if highlightAtoms is not None and not isinstance(highlightAtoms, list):
        raise ValueError(
            "highlightAtoms should be a list of atom idx or a list of atom idx list "
        )

    ext = ".svg"
    if filename is not None:
        ext = splitext(filename)[-1]
        if ext == "":
            ext = ".svg"
            filename = filename + ".svg"

    # init the drawer object
    if ext == ".png":
        drawer = rdMolDraw2D.MolDraw2DCairo(*resolution)
    elif ext == ".svg":
        drawer = rdMolDraw2D.MolDraw2DSVG(*resolution)
    else:
        raise RuntimeError(
            f"Unsupported depiction extention {ext}. Use either svg or png."
        )
    # get the drawer options
    opts = drawer.drawOptions()

    # add atomlabels
    if atomlabels is not None:
        for n, a in enumerate(atomlabels):
            opts.atomLabels[n] = a

    # highlight atoms
    sel_colors = {}
    if highlightAtoms is not None:
        if not isinstance(highlightAtoms[0], list):
            highlightAtoms = [highlightAtoms]

        for n, subset in enumerate(highlightAtoms):
            for aIdx in subset:
                if aIdx not in sel_colors:
                    sel_colors[aIdx] = []
                sel_colors[aIdx].append(_highlight_colors[n % len(_highlight_colors)])

    Kekulize(mol)
    if np.any([len(sel_colors[aIdx]) > 1 for aIdx in sel_colors]):
        drawer.drawOptions().fillHighlights = False
    drawer.DrawMoleculeWithHighlights(mol, "", sel_colors, {}, {}, {}, -1)
    drawer.FinishDrawing()

    # svg object
    svg = drawer.GetDrawingText()

    # activate saving into a file
    if filename is not None:
        if ext == ".svg":
            f = open(filename, "w")
        elif ext == ".png":
            f = open(filename, "wb")
        f.write(svg)
        f.close()

    # activate jupiter-notebook rendering
    if ipython:
        svg = svg.replace("svg:", "")
        return SVG(svg)
    else:
        return None


def depictMultipleMols(
    mols_list,
    filename=None,
    ipython=False,
    legends=None,
    highlightAtoms=None,
    mols_perrow=3,
):
    """
    Returns the image or the ipython rendering.

    Parameters
    ----------
    mols_list: list
        The list of the rdkit molecules to depict
    filename: str
        The filename of the image
    ipython: bool
        If True, the SVG rendering for jupiter-nootebook are returned
    legends: list
        List of titles subfigure for each molecule
    highlightAtoms: list
        List of list of atom index to highlight.
    mols_perrow: int
        The number of subfigures per row

    Returns
    -------
    svg: SVG
        If ipython set as True, the SVG rendering is returned

    """
    from rdkit.Chem.Draw import MolsToGridImage
    from IPython.display import SVG
    from os.path import splitext

    sel_atoms = []
    sel_colors = []
    if highlightAtoms is not None:
        if isinstance(highlightAtoms[0][0], list):
            sel_atoms = [
                [a for a in subset] for mol_set in highlightAtoms for subset in mol_set
            ]
            sel_colors = [
                {aIdx: _highlight_colors[n % len(_highlight_colors)] for aIdx in subset}
                for mol_set in highlightAtoms
                for n, subset in enumerate(mol_set)
            ]
        else:
            sel_atoms = highlightAtoms
            sel_colors = [
                {aIdx: _highlight_colors[0] for aIdx in subset}
                for subset in highlightAtoms
            ]

    from rdkit.Chem.Draw import IPythonConsole as CDIPythonConsole

    if MolsToGridImage == CDIPythonConsole.ShowMols:
        CDIPythonConsole.UninstallIPythonRenderer()
        from rdkit.Chem.Draw import MolsToGridImage

    svg = MolsToGridImage(
        mols_list,
        highlightAtomLists=sel_atoms,
        highlightBondLists=[],
        highlightAtomColors=sel_colors,
        legends=legends,
        molsPerRow=mols_perrow,
        useSVG=True,
    )

    if filename:
        ext = splitext(filename)[-1]
        filename = filename if ext != "" else filename + ".svg"
        f = open(filename, "w")
        f.write(svg)
        f.close()

    if ipython:
        _svg = SVG(svg)
        return _svg
    else:
        return None
