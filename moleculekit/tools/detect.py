# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import logging
from collections import OrderedDict
import itertools
import networkx as nx
from moleculekit.periodictable import periodictable
import unittest

logger = logging.getLogger(__name__)


def _getMolecularGraph(molecule):
    """
    Generate a graph from the topology of molecule

    The graph nodes represent atoms and the graph edges represent bonds. Also, the graph nodes store element
    information.
    """

    graph = nx.Graph()
    for i, element in enumerate(molecule.element):
        graph.add_node(
            i, element=element, number=periodictable[element.capitalize()].number
        )
    graph.add_edges_from(molecule.bonds)

    return graph


def _getMolecularTree(graph, source):
    """
    Generate a tree from a molecular graph

    The tree starts from source node (atom) and grows along the edges (bonds) unrolling all encountered loops.
    The tree grows until all the nodes (atoms) of the graph are included.
    """

    assert nx.is_connected(graph)

    tree = nx.DiGraph()
    tree.add_node(0, base=source, element=graph.nodes[source]["element"])
    current_nodes = list(tree.nodes)
    base_nodes = {source}

    while True:
        new_nodes = []
        neighbor_filter = lambda node: node not in base_nodes
        for current_node in current_nodes:
            for neighbor in filter(
                neighbor_filter, graph.neighbors(tree.nodes[current_node]["base"])
            ):
                new_node = len(tree.nodes)
                tree.add_node(
                    new_node, base=neighbor, element=graph.nodes[neighbor]["element"]
                )
                tree.add_edge(current_node, new_node)
                new_nodes.append(new_node)

        current_nodes = new_nodes
        base_nodes = {base for _, base in tree.nodes.data("base")}
        if base_nodes == set(graph.nodes):
            break

    return tree


def _checkIsomorphism(graph1, graph2):
    """
    Check if two molecular graphs are isomorphic based on topology (bonds) and elements.
    """

    return nx.is_isomorphic(
        graph1,
        graph2,
        node_match=lambda node1, node2: node1["element"] == node2["element"],
    )


def detectEquivalentAtoms(molecule):
    """
    Detect topologically equivalent atoms.

    Arguments
    ---------
    molecule : :class:`Molecule <moleculekit.molecule.Molecule>`
        Molecule object

    Return
    ------
    equivalent_groups : list of tuples
        List of equivalent atom group. Each element is a tuple contain equivalent atom indices.
    equivalent_atoms : list of tuples
        List of equivalent atom group for each atom. Each element is a tuple contain equivalent atom indices.
    equivalent_group_by_atom : list
        List of equivalent group indices for each atom. The indices corresponds to `equivalent_groups` order.

    Examples
    --------

    >>> import os
    >>> from moleculekit.home import home
    >>> from moleculekit.molecule import Molecule
    >>> from moleculekit.tools.detect import detectEquivalentAtoms

    Get benzamidine
    >>> molFile = os.path.join(home('test-detect'), 'benzamidine.mol2')
    >>> mol = Molecule(molFile)

    Find the equivalent atoms of bezamidine
    >>> equivalent_groups, equivalent_atoms, equivalent_group_by_atom = detectEquivalentAtoms(mol)
    >>> equivalent_groups
    [(0,), (1, 5), (2, 4), (3,), (6,), (7, 11), (8, 10), (9,), (12, 13), (14, 15, 16, 17)]
    >>> equivalent_atoms
    [(0,), (1, 5), (2, 4), (3,), (2, 4), (1, 5), (6,), (7, 11), (8, 10), (9,), (8, 10), (7, 11), (12, 13), (12, 13), (14, 15, 16, 17), (14, 15, 16, 17), (14, 15, 16, 17), (14, 15, 16, 17)]
    >>> equivalent_group_by_atom
    [0, 1, 2, 3, 2, 1, 4, 5, 6, 7, 6, 5, 8, 8, 9, 9, 9, 9]

    Get dicarbothioic acid
    >>> molFile = os.path.join(home('test-detect'), 'dicarbothioic_acid.mol2')
    >>> mol = Molecule(molFile)

    Find the equivalent atoms of dicarbothioic acid
    >>> equivalent_groups, equivalent_atoms, equivalent_group_by_atom = detectEquivalentAtoms(mol)
    >>> equivalent_groups
    [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)]
    >>> equivalent_atoms
    [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)]
    >>> equivalent_group_by_atom
    [0, 1, 2, 3, 4, 5, 6, 7]
    """
    from networkx.algorithms.isomorphism import rooted_tree_isomorphism

    # Generate a molecular tree for each atom
    graph = _getMolecularGraph(molecule)
    trees = [_getMolecularTree(graph, node) for node in graph.nodes]

    equivalent_atoms = {}
    equivalent_groups = [[0]]  # Start first group with node 0
    equivalent_atoms[0] = equivalent_groups[0]
    for i in range(1, len(trees)):
        for g, grp in enumerate(equivalent_groups):
            if _checkIsomorphism(trees[i], trees[grp[0]]):
                equivalent_groups[g].append(i)
                equivalent_atoms[i] = equivalent_groups[g]
                break
        else:
            equivalent_groups.append([i])
            equivalent_atoms[i] = equivalent_groups[-1]

    equivalent_groups = [tuple(sorted(eg)) for eg in equivalent_groups]
    equivalent_atoms = [tuple(equivalent_atoms[i]) for i in range(len(trees))]
    equivalent_group_by_atom = list(map(equivalent_groups.index, equivalent_atoms))

    return equivalent_groups, equivalent_atoms, equivalent_group_by_atom


def _getMethylGraph():
    """
    Generate a molecular graph for methyl group
    """

    methyl = nx.Graph()
    methyl.add_node(0, element="C")
    methyl.add_node(1, element="H")
    methyl.add_node(2, element="H")
    methyl.add_node(3, element="H")
    methyl.add_edge(0, 1)
    methyl.add_edge(0, 2)
    methyl.add_edge(0, 3)

    return methyl


def connected_component_subgraphs(graph):
    return (
        graph.subgraph(component).copy() for component in nx.connected_components(graph)
    )


def detectParameterizableCores(graph):
    """
    Detect parametrizable dihedral angle cores (central atom pairs)

    The cores are detected by looking for bridges (bonds which divide the molecule into two parts) in a molecular graph.
    Terminal cores are skipped.
    """

    methyl = _getMethylGraph()

    all_core_sides = []
    for core in list(nx.bridges(graph)):

        # Get side graphs of the core
        graph.remove_edge(*core)
        sideGraphs = list(connected_component_subgraphs(graph))
        graph.add_edge(*core)

        # Skip terminal bridges, which cannot form dihedral angles
        if len(sideGraphs[0]) == 1 or len(sideGraphs[1]) == 1:
            continue

        # Swap the side graphs to match the order of the core
        sideGraphs = sideGraphs[::-1] if core[0] in sideGraphs[1] else sideGraphs
        assert core[0] in sideGraphs[0] and core[1] in sideGraphs[1]

        # Skip if a side graph is a methyl group
        if _checkIsomorphism(sideGraphs[0], methyl) or _checkIsomorphism(
            sideGraphs[1], methyl
        ):
            continue

        # Skip if core contains C with sp hybridization
        if graph.nodes[core[0]]["element"] == "C" and graph.degree(core[0]) == 2:
            continue
        if graph.nodes[core[1]]["element"] == "C" and graph.degree(core[1]) == 2:
            continue

        all_core_sides.append((core, sideGraphs))

    return all_core_sides


def _weighted_closeness_centrality(graph, node, weight=None):
    """
    Weighted closeness centrality

    Identical to networkx.closeness_centrality, except the shorted path lengths are weighted by a node attribute.
    """

    lengths = nx.shortest_path_length(graph, source=node)
    del lengths[node]
    weights = (
        {node_: graph.nodes[node_][weight] for node_ in lengths}
        if weight
        else {node_: 1 for node_ in lengths}
    )
    centrality = sum(weights.values()) / sum(
        [lengths[node_] * weights[node_] for node_ in lengths]
    )

    return centrality


def _chooseTerminals(graph, centre, sideGraph):
    """
    Choose dihedral angle terminals (outer atoms)

    The terminals are chosen by:
    1. Largest closeness centrality
    2. Largest atomic number weighted closeness centrality
    """

    terminals = list(sideGraph.neighbors(centre))

    # Get a subgraph for each terminal
    sideGraph = sideGraph.copy()
    sideGraph.remove_node(centre)
    terminalGraphs = itertools.product(
        terminals, connected_component_subgraphs(sideGraph)
    )
    terminalGraphs = [
        terminalGraph
        for terminal, terminalGraph in terminalGraphs
        if terminal in terminalGraph
    ]

    # Compute a score for each terminal
    centralities = [nx.closeness_centrality(graph, terminal) for terminal in terminals]
    weightedCentralities = [
        _weighted_closeness_centrality(graph, terminal, weight="number")
        for terminal in terminals
    ]
    scores = list(zip(centralities, weightedCentralities))

    # Choose the terminals
    chosen_terminals = []
    refTerminalGraph = None
    for terminal, score, terminalGraph in zip(terminals, scores, terminalGraphs):
        if score < max(scores):
            continue

        if not chosen_terminals:
            chosen_terminals.append(terminal)
            refTerminalGraph = terminalGraph
            continue

        if _checkIsomorphism(terminalGraph, refTerminalGraph):
            chosen_terminals.append(terminal)
        else:
            logger.warn(
                "Molecular scoring function is not sufficient. "
                "Dihedal selection depends on the atom order! "
                "Redundant dihedrals might be present!"
            )

    return chosen_terminals


def detectParameterizableDihedrals(molecule, exclude_atoms=()):
    """
    Detect parameterizable dihedral angles

    Arguments
    ---------
    molecule : :class:`Molecule <moleculekit.molecule.Molecule>`
        Molecule object
    exclude_atoms : list
        Ignore dihedrals which consist purely of atoms in this list

    Return
    ------
    dihedrals : list of list of tuples
        List of equivalent dihedral angle groups. Each group is a list of equivalent dihedral angles.
        Each angle is defined as a tuple of four atom indices (0-based).

    Examples
    --------

    >>> import os
    >>> from moleculekit.home import home
    >>> from moleculekit.molecule import Molecule
    >>> from moleculekit.tools.detect import detectParameterizableDihedrals

    Find the parameterizable dihedrals of glycol
    >>> molFile = os.path.join(home('test-detect'), 'glycol.mol2')
    >>> mol = Molecule(molFile, guess=('bonds', 'angles', 'dihedrals'))
    >>> detectParameterizableDihedrals(mol)
    [[(0, 1, 2, 3)], [(1, 2, 3, 9), (2, 1, 0, 4)]]

    Find the parameterizable dihedrals of ethanolamine
    >>> molFile = os.path.join(home('test-detect'), 'ethanolamine.mol2')
    >>> mol = Molecule(molFile, guess=('bonds', 'angles', 'dihedrals'))
    >>> detectParameterizableDihedrals(mol)
    [[(0, 1, 2, 3)], [(1, 2, 3, 9), (1, 2, 3, 10)], [(2, 1, 0, 4)]]

    Find the parameterizable dihedrals of benzamidine
    >>> molFile = os.path.join(home('test-detect'), 'benzamidine.mol2')
    >>> mol = Molecule(molFile, guess=('bonds', 'angles', 'dihedrals'))
    >>> detectParameterizableDihedrals(mol)
    [[(0, 6, 12, 16), (0, 6, 12, 17), (0, 6, 13, 14), (0, 6, 13, 15)], [(1, 0, 6, 12), (1, 0, 6, 13), (5, 0, 6, 12), (5, 0, 6, 13)]]

    # Check if the atom swapping does not affect results

    Find the parameterizable dihedrals of chlorethene
    >>> molFile = os.path.join(home('test-detect'), 'chlorethene_1.mol2')
    >>> mol = Molecule(molFile, guess=('bonds', 'angles', 'dihedrals'))
    >>> detectParameterizableDihedrals(mol)
    [[(2, 1, 0, 4), (2, 1, 0, 5)]]

    Find the parameterizable dihedrals of chlorethene (with swapped atoms)
    >>> molFile = os.path.join(home('test-detect'), 'chlorethene_2.mol2')
    >>> mol = Molecule(molFile, guess=('bonds', 'angles', 'dihedrals'))
    >>> detectParameterizableDihedrals(mol)
    [[(3, 1, 0, 4), (3, 1, 0, 5)]]

    # Check if triple bonds are skipped

    Find the parameterizable dihedrals of 4-hexinenitrile
    >>> molFile = os.path.join(home('test-detect'), '4-hexinenitrile.mol2')
    >>> mol = Molecule(molFile, guess=('bonds', 'angles', 'dihedrals'))
    >>> detectParameterizableDihedrals(mol)
    [[(2, 3, 4, 5)]]

    # Check the scoring function

    Find the parameterizable dihedrals of dicarbothioic acid
    >>> molFile = os.path.join(home('test-detect'), 'dicarbothioic_acid.mol2')
    >>> mol = Molecule(molFile, guess=('bonds', 'angles', 'dihedrals'))
    >>> detectParameterizableDihedrals(mol)
    [[(0, 1, 3, 5)], [(1, 3, 5, 7)], [(3, 1, 0, 6)]]

    Find the parameterizable dihedrals of 2-hydroxypyridine
    >>> molFile = os.path.join(home('test-detect'), '2-hydroxypyridine.mol2')
    >>> mol = Molecule(molFile, guess=('bonds', 'angles', 'dihedrals'))
    >>> detectParameterizableDihedrals(mol)
    [[(6, 1, 0, 7)]]

    Find the parameterizable dihedrals of fluorchlorcyclopronol
    >>> molFile = os.path.join(home('test-detect'), 'fluorchlorcyclopronol.mol2')
    >>> mol = Molecule(molFile, guess=('bonds', 'angles', 'dihedrals'))
    >>> detectParameterizableDihedrals(mol)
    [[(2, 4, 5, 9)]]
    """
    # Get a molecular graph
    graph = _getMolecularGraph(molecule)

    # Get parameterizable dihedral angles
    dihedrals = []
    for core, sides in detectParameterizableCores(graph):

        # Choose the best terminals for each side
        all_terminals = [
            _chooseTerminals(graph, centre, side) for centre, side in zip(core, sides)
        ]

        # Generate all terminal combinations
        all_terminals = itertools.product(*all_terminals)

        # Generate new dihedral angles
        for terminals in all_terminals:
            new_dihedral = (terminals[0], *core, terminals[1])
            if all([idx in exclude_atoms for idx in new_dihedral]):
                # Skip dihedrals consisting purely of atoms in exclude_atoms
                continue
            dihedrals.append(new_dihedral)

    # Get equivalent groups for each atom for each dihedral
    _, _, equivalent_group_by_atom = detectEquivalentAtoms(molecule)
    dihedral_groups = [
        tuple([equivalent_group_by_atom[atom] for atom in dihedral])
        for dihedral in dihedrals
    ]

    # Group equivalent dihedral angles and reverse them that equivalent atoms are matched
    equivalent_dihedrals = OrderedDict()
    for dihedral, groups in zip(dihedrals, dihedral_groups):
        dihedral, groups = (
            (dihedral[::-1], groups[::-1])
            if groups[::-1] < groups
            else (dihedral, groups)
        )
        equivalent_dihedrals[groups] = sorted(
            equivalent_dihedrals.get(groups, []) + [dihedral]
        )
    equivalent_dihedrals = sorted(equivalent_dihedrals.values())
    return equivalent_dihedrals


class _TestEquivDetection(unittest.TestCase):
    def test_atom_detection(self):
        import os
        from moleculekit.home import home
        from moleculekit.molecule import Molecule
        import numpy as np

        mol = Molecule(os.path.join(home(dataDir="test-detect"), "KCX.cif"))
        eqgroups, eqatoms, eqgroupbyatom = detectEquivalentAtoms(mol)

        # fmt: off
        eqgroups_ref = [(0, 2, 3), (1,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11, 12, 13), (14,), (15,), (16,), (17,), (18,), (19,), (20,), (21,), (22,), (23,), (24,), (25,), (26, 27), (28, 29), (30, 31), (32, 33), (34, 35), (36,), (37,), (38,), (39,), (40,), (41,), (42,), (43,), (44, 45, 46), (47,), (48,), (49,), (50,), (51,), (52, 53, 54)]
        eqatoms_ref = [(0, 2, 3), (1,), (0, 2, 3), (0, 2, 3), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11, 12, 13), (11, 12, 13), (11, 12, 13), (14,), (15,), (16,), (17,), (18,), (19,), (20,), (21,), (22,), (23,), (24,), (25,), (26, 27), (26, 27), (28, 29), (28, 29), (30, 31), (30, 31), (32, 33), (32, 33), (34, 35), (34, 35), (36,), (37,), (38,), (39,), (40,), (41,), (42,), (43,), (44, 45, 46), (44, 45, 46), (44, 45, 46), (47,), (48,), (49,), (50,), (51,), (52, 53, 54), (52, 53, 54), (52, 53, 54)]
        eqgroupbyatom_ref = [0, 1, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 35, 35, 36, 37, 38, 39, 40, 41, 41, 41]
        # fmt: on
        assert np.array_equal(
            np.array(eqgroups, dtype=object), np.array(eqgroups_ref, dtype=object)
        )
        assert np.array_equal(
            np.array(eqatoms, dtype=object), np.array(eqatoms_ref, dtype=object)
        )
        assert np.array_equal(
            np.array(eqgroupbyatom, dtype=object),
            np.array(eqgroupbyatom_ref, dtype=object),
        )


if __name__ == "__main__":

    import sys
    import doctest

    # Prevent HTMD importing inside doctest to fail if importing gives text output
    from moleculekit.home import home

    home()

    if doctest.testmod().failed:
        sys.exit(1)

    unittest.main(verbosity=2)
