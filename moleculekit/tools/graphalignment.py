# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
try:
    import networkx as nx
except ImportError:
    print(
        "Could not import networkx which is necessary for graph alignment. You can install it with `conda install networkx`."
    )
from moleculekit.util import ensurelist
from unittest import TestCase
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


def createProductGraph(G, H, tolerance, fields):
    # Calculate product graph by creating a node for each feature-matching pair of points
    newnodes = []
    for gn in G.nodes():
        for hn in H.nodes():
            matching = True
            for f in fields:
                if G.nodes[gn][f] != H.nodes[hn][f]:
                    matching = False
                    break
            if matching:
                newnodes.append((gn, hn))

    Gprod = nx.Graph()
    Gprod.add_nodes_from(newnodes)

    # Add edges when distances in both graphs agree within tolerance error between the two given nodes
    for np1 in range(len(newnodes)):
        for np2 in range(np1 + 1, len(newnodes)):
            pair1 = newnodes[np1]
            pair2 = newnodes[np2]
            if not G.has_edge(pair1[0], pair2[0]) or not H.has_edge(pair1[1], pair2[1]):
                continue
            dist1 = G.edges[pair1[0], pair2[0]]["distance"]
            dist2 = H.edges[pair1[1], pair2[1]]["distance"]
            if abs(dist1 - dist2) < tolerance:
                Gprod.add_edge(newnodes[np1], newnodes[np2])
    return Gprod


def compareGraphs(G, H, fields=("element",), tolerance=0.5, returnmatching=False):
    # Comparison algorithm based on:
    # "Chemoisosterism in the Proteome", X. Jalencas, J. Mestres, JCIM 2013
    # http://pubs.acs.org/doi/full/10.1021/ci3002974
    fields = ensurelist(fields)
    if G == H:
        if returnmatching:
            return 1, len(G), [(x, x) for x in G.nodes()]
        else:
            return 1

    if len(G.edges()) == 0 or len(H.edges()) == 0:
        if returnmatching:
            return 0, 0, []
        else:
            return 0

    Gprod = createProductGraph(G, H, tolerance, fields)

    # Calculate the maximal cliques and return the length of the largest one
    maxcliques = list(nx.find_cliques(Gprod))
    cllen = np.array([len(x) for x in maxcliques])
    score = cllen.max() / max(len(G.nodes()), len(H.nodes()))

    if returnmatching:
        return score, cllen.max(), maxcliques[cllen.argmax()]
    else:
        return score


def makeMolGraph(mol, sel, fields):
    from scipy.spatial.distance import pdist, squareform

    if sel != "all":
        sel = mol.atomselect(sel, indexes=True)
    else:
        sel = np.arange(mol.numAtoms)

    g = nx.Graph()
    for i in sel:
        props = {f: mol.__dict__[f][i] for f in fields}
        g.add_node(i, **props)

    distances = squareform(pdist(mol.coords[sel, :, mol.frame]))
    nodes = list(g.nodes())
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            g.add_edge(nodes[i], nodes[j], distance=distances[i, j])

    return g


def maximalSubstructureAlignment(
    mol1,
    mol2,
    sel1="all",
    sel2="all",
    fields=("element",),
    tolerance=0.5,
    visualize=False,
):
    """Aligns two molecules on the largest common substructure

    Parameters
    ----------
    mol1 : :class:`Molecule`
        The reference molecule on which to align
    mol2 : :class:`Molecule`
        The second molecule which will be rotated and translated to align on mol1
    sel1 : str
        Atom selection string of the atoms of `mol1` to align.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    sel2 : str
        Atom selection string of the atoms of `mol2` to align.
        See more `here <http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/ug/node89.html>`__
    fields : tuple
        A tuple of the fields that are used to match atoms
    tolerance : float
        How different can distances be between to atom pairs for them to match in the product graph
    visualize : bool
        If set to True it will visualize the alignment

    Returns
    -------
    newmol : :class:`Molecule`
        A copy of mol2 aligned on mol1
    """
    mol2 = mol2.copy()
    g1 = makeMolGraph(mol1, sel1, fields)
    g2 = makeMolGraph(mol2, sel2, fields)

    _, _, matching = compareGraphs(
        g1, g2, fields=fields, tolerance=tolerance, returnmatching=True
    )

    matchnodes1 = np.array([x[0] for x in matching])
    matchnodes2 = np.array([x[1] for x in matching])

    mol2.align(sel=matchnodes2, refmol=mol1, refsel=matchnodes1)

    if visualize:
        mol1.view(
            sel=f"index {' '.join(map(str, matchnodes1))}",
            style="CPK",
            hold=True,
        )
        mol1.view(sel="all", style="Lines")

        mol2.view(
            sel=f"index {' '.join(map(str, matchnodes2))}",
            style="CPK",
            hold=True,
        )
        mol2.view(sel="all", style="Lines")

    return mol2


def mcsAtomMatching(
    mol1, mol2, atomCompare="elements", bondCompare="any", _logger=True
):
    """Maximum common substructure atom matching.

    Given two molecules it will find their maximum common substructure using rdkit
    and return the atoms in both molecules which matched.

    Parameters
    ----------
    mol1 : Molecule
        The first molecule
    mol2 : Molecule
        The second molecule
    atomCompare : str
        Which features of the atoms to compare. Can be either: "any", "elements" or "isotopes"
    bondCompare : str
        Which features of the bonds to compare. Can be either: "any", "order" or "orderexact"

    Returns
    -------
    atm1 : list
        A list of atom indexes of the first molecule which matched to the second
    atm2 : list
        A list of atom indexes of the second molecule which matched to the first

    Examples
    --------
    >>> mol1 = Molecule("OIC.cif")
    >>> mol1.atomtype = mol1.element
    >>> mol2 = Molecule("5vbl")
    >>> mol2.filter("resname OIC")
    >>> atm1, atm2 = mcsAtomMatching(mol1, mol2, bondCompare="any")
    >>> print(mol1.name[atm1], mol2.name[atm2])
    ['N' 'CA' 'C' 'O' 'CB' 'CG' 'CD' 'C7' 'C6' 'C5' 'C4'] ['N' 'CA' 'C' 'O' 'CB' 'CG' 'CD' 'C7' 'C6' 'C5' 'C4']
    """
    from moleculekit.smallmol.smallmol import SmallMol
    from rdkit.Chem import rdFMCS
    from rdkit import Chem

    atmcmp = {
        "any": rdFMCS.AtomCompare.CompareAny,
        "elements": rdFMCS.AtomCompare.CompareElements,
        "isotopes": rdFMCS.AtomCompare.CompareIsotopes,
    }
    atmcmp = atmcmp[atomCompare.lower()]

    bndcmp = {
        "any": rdFMCS.BondCompare.CompareAny,
        "order": rdFMCS.BondCompare.CompareOrder,
        "orderexact": rdFMCS.BondCompare.CompareOrderExact,
    }
    bndcmp = bndcmp[bondCompare.lower()]
    if bondCompare.lower() != "any":
        if np.any(np.isin(mol1.bondtype, ("", "un"))) or np.any(
            np.isin(mol2.bondtype, ("", "un"))
        ):
            raise RuntimeError(
                f"Using mcsAtomMatching with bondCompare {bondCompare} requires bond orders in the molecules"
            )

    smol1 = SmallMol(mol1, fixHs=False, removeHs=False)._mol
    smol2 = SmallMol(mol2, fixHs=False, removeHs=False)._mol
    res = rdFMCS.FindMCS([smol1, smol2], bondCompare=bndcmp, atomCompare=atmcmp)
    patt = Chem.MolFromSmarts(res.smartsString)
    at1 = list(smol1.GetSubstructMatch(patt))
    at2 = list(smol2.GetSubstructMatch(patt))

    if _logger:
        n_heavy1 = np.sum(mol1.element != "H")
        m_heavy1 = np.sum(mol1.element[at1] != "H")
        n_heavy2 = np.sum(mol2.element != "H")
        m_heavy2 = np.sum(mol2.element[at2] != "H")
        n_atm1 = mol1.numAtoms
        n_atm2 = mol2.numAtoms
        msg = f"Matched {m_heavy1}/{n_heavy1} heavy atoms in mol1 to {m_heavy2}/{n_heavy2} heavy atoms in mol2."
        if np.any(mol1.element == "H") and np.any(mol2.element == "H"):
            msg += f" Matched {len(at1)-m_heavy1}/{n_atm1-n_heavy1} hydrogens in mol1 to {len(at2)-m_heavy2}/{n_atm2-n_heavy2} hydrogens in mol2."
        logger.info(msg)
    return at1, at2


class _TestGraphAlignment(TestCase):
    def test_maximalSubstructureAlignment(self):
        from moleculekit.home import home
        from moleculekit.molecule import Molecule

        path = home(dataDir="test-molecule-graphalignment")
        ref_lig = Molecule(os.path.join(path, "ref_lig.pdb"))
        lig2align = Molecule(os.path.join(path, "lig2align.pdb"))
        lig_aligned = maximalSubstructureAlignment(ref_lig, lig2align)
        lig_reference = Molecule(os.path.join(path, "lig_aligned.pdb"))

        self.assertTrue(
            np.allclose(lig_aligned.coords, lig_reference.coords, rtol=1e-4),
            "maximalSubstructureAlignment produced different coords",
        )

    def test_mcs_atom_matching(self):
        from moleculekit.home import home
        from moleculekit.molecule import Molecule
        import numpy as np

        path = home(dataDir="test-molecule-graphalignment")

        mol1 = Molecule(os.path.join(path, "OIC.cif"))
        mol1.atomtype = mol1.element
        idx = np.random.permutation(np.arange(mol1.numAtoms))
        mol1.reorderAtoms(idx)

        mol2 = Molecule("5VBL")
        mol2.filter("resname OIC")
        atm1, atm2 = mcsAtomMatching(mol1, mol2, bondCompare="order")
        assert len(atm1) == 11
        assert np.array_equal(mol1.name[atm1], mol2.name[atm2])

        atm1, atm2 = mcsAtomMatching(mol1, mol2, bondCompare="any")
        assert len(atm1) == 11
        # Compare elements here since it can match O to OXT when ignoring bond type
        assert np.array_equal(mol1.element[atm1], mol2.element[atm2])


if __name__ == "__main__":
    import unittest

    unittest.main(verbosity=2)
