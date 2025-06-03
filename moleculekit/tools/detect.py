# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import logging
import itertools
import networkx as nx
from moleculekit.periodictable import periodictable

logger = logging.getLogger(__name__)


def _getMolecularGraph(mol):
    """
    Generate a graph from the topology of molecule

    The graph nodes represent atoms and the graph edges represent bonds. Also, the graph nodes store element
    information.
    """

    graph = nx.Graph()
    for i in range(mol.numAtoms):
        graph.add_node(
            i,
            element=mol.element[i],
            number=periodictable[mol.element[i].capitalize()].number,
            formalcharge=mol.formalcharge[i],
        )
    graph.add_edges_from(mol.bonds)

    return graph


def _getMolecularTree(graph, source):
    """
    Generate a tree from a molecular graph

    The tree starts from source node (atom) and grows along the edges (bonds) unrolling all encountered loops.
    The tree grows until all the nodes (atoms) of the graph are included.
    """

    assert nx.is_connected(graph)

    tree = nx.DiGraph()
    tree.add_node(0, base=source, **graph.nodes[source])
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
                tree.add_node(new_node, base=neighbor, **graph.nodes[neighbor])
                tree.add_edge(current_node, new_node)
                new_nodes.append(new_node)

        current_nodes = new_nodes
        base_nodes = {base for _, base in tree.nodes.data("base")}
        if base_nodes == set(graph.nodes):
            break

    return tree


def rooted_tree_isomorphism(t1, root1, t2, root2, keyfunc):
    """Return if two rooted trees are isomorphic

    Given two rooted trees `t1` and `t2`,
    with roots `root1` and `root2` respectively
    this routine will determine if they are isomorphic.

    The idea is that for each node we build a hash of the node and its children which takes into account
    all the properties of the node which we want (here elements and formal charges).
    Thus isomorphic trees will have identical hashes.

    We start from the bottom of the tree, going up. Everytime we find a new node+children combination
    we generate a new hash.
    If at any level hashes don't match it means that one of the two trees has a different structure,
    this allows us to quit early while going up the tree.

    The idea on how to iterate the tree is based on the same named networkx function, but the algorithm is completely original.

    Parameters
    ----------
    t1 :  NetworkX graph
        One of the trees being compared
    root1 : int
        a node of `t1` which is the root of the tree
    t2 : undirected NetworkX graph
        The other tree being compared
    root2 : int
        a node of `t2` which is the root of the tree
    keyfunc : function
        Function which generates the description key for each node.
        It should take as input the node and return a string
    """
    import numpy as np

    # figure out the level of each node, with 0 at root
    def assign_levels(G, root):
        level = {}
        level[root] = 0
        for v1, v2 in nx.bfs_edges(G, root):
            level[v2] = level[v1] + 1

        return level

    # now group the nodes at each level
    def group_by_levels(levels):
        L = {}
        for n, lev in levels.items():
            if lev not in L:
                L[lev] = []
            L[lev].append(n)

        return L

    assert nx.is_tree(t1)
    assert nx.is_tree(t2)

    # Check if root nodes have same key
    if keyfunc(t1.nodes[root1]) != keyfunc(t2.nodes[root2]):
        return False

    # compute the distance from the root, with 0 for our
    levels1 = assign_levels(t1, root1)
    levels2 = assign_levels(t2, root2)

    # height
    h1 = max(levels1.values())
    h2 = max(levels2.values())
    if h1 != h2:
        return False

    # collect nodes into a dict by level
    L1 = group_by_levels(levels1)
    L2 = group_by_levels(levels2)

    # We add all atom properties to the atom_props dictionaries
    # The hashes dict stores the unique hashes which are discovered while traversing the tree
    # node_hashes stores the final hashes assigned to each node. We start off by giving them the hash of their element/charge
    node_hashes1 = {}
    node_hashes2 = {}
    atom_props1 = {}
    atom_props2 = {}
    hashes = {}
    hashcount = 0
    for v in t1:
        pp = t1.nodes[v]
        key = keyfunc(pp)
        atom_props1[v] = key
        if key not in hashes:
            hashes[key] = hashcount
            hashcount += 1
        node_hashes1[v] = hashes[key]

    for v in t2:
        pp = t2.nodes[v]
        key = keyfunc(pp)
        atom_props2[v] = key
        if key not in hashes:
            hashes[key] = hashcount
            hashcount += 1
        node_hashes2[v] = hashes[key]

    # nothing to do on last level so start on h-1
    for i in range(h1 - 1, -1, -1):
        if len(L1[i]) != len(L2[i]):
            return False

        curr_hashes1 = []
        curr_hashes2 = []

        for v in L1[i]:
            node_key = atom_props1[v]
            if t1.out_degree(v) > 0:
                # Create a new label for the atom which looks like X|Y,Z,W where X is the hash
                # of the current atom and Y,Z,W the hashes of it's i.e. 3 children
                node_key = (
                    str(node_hashes1[v])
                    + "|"
                    + ",".join(sorted(str(node_hashes1[u]) for u in t1.successors(v)))
                )
                if node_key not in hashes:
                    hashes[node_key] = hashcount
                    hashcount += 1
            # Assign to this node the new hash
            node_hashes1[v] = hashes[node_key]
            # Keep a collection of the hashes of this level to compare at the end
            curr_hashes1.append(hashes[node_key])

        # Identical loop as above for the second tree. Too lazy to generalize it right now
        # TODO: Generalize this loop out
        for v in L2[i]:
            node_key = atom_props2[v]
            if t2.out_degree(v) > 0:
                # get all the pairs of labels and nodes of children
                # and sort by labels
                node_key = (
                    str(node_hashes2[v])
                    + "|"
                    + ",".join(sorted(str(node_hashes2[u]) for u in t2.successors(v)))
                )
                if node_key not in hashes:
                    hashes[node_key] = hashcount
                    hashcount += 1
            node_hashes2[v] = hashes[node_key]
            curr_hashes2.append(hashes[node_key])

        # If the hashes of the current level don't match, the trees are not isomorphic
        if not np.array_equal(sorted(curr_hashes1), sorted(curr_hashes2)):
            return False

    # If the root hashes are identical we won! The trees are isomorphic
    return node_hashes1[root1] == node_hashes2[root2]


def _checkIsomorphism(graph1, graph2):
    """
    Check if two molecular graphs are isomorphic based on topology (bonds) and elements.
    """

    return nx.is_isomorphic(
        graph1,
        graph2,
        node_match=lambda node1, node2: node1["element"] == node2["element"]
        and node1["formalcharge"] == node2["formalcharge"],
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
    >>> molFile = 'benzamidine.mol2'
    >>> mol = Molecule(molFile)

    Find the equivalent atoms of benzamidine
    >>> equivalent_groups, equivalent_atoms, equivalent_group_by_atom = detectEquivalentAtoms(mol)
    >>> equivalent_groups
    [(0,), (1, 5), (2, 4), (3,), (6,), (7, 11), (8, 10), (9,), (12, 13), (14, 15, 16, 17)]
    >>> equivalent_atoms
    [(0,), (1, 5), (2, 4), (3,), (2, 4), (1, 5), (6,), (7, 11), (8, 10), (9,), (8, 10), (7, 11), (12, 13), (12, 13), (14, 15, 16, 17), (14, 15, 16, 17), (14, 15, 16, 17), (14, 15, 16, 17)]
    >>> equivalent_group_by_atom
    [0, 1, 2, 3, 2, 1, 4, 5, 6, 7, 6, 5, 8, 8, 9, 9, 9, 9]

    Get dicarbothioic acid
    >>> molFile = 'dicarbothioic_acid.mol2'
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
    # Generate a molecular tree for each atom
    graph = molecule.toGraph(fields=("element", "formalcharge"))
    trees = [_getMolecularTree(graph, node) for node in graph.nodes]

    equivalent_atoms = {}
    equivalent_groups = [[0]]  # Start first group with node 0
    equivalent_atoms[0] = equivalent_groups[0]
    for i in range(1, len(trees)):
        for g, grp in enumerate(equivalent_groups):
            if rooted_tree_isomorphism(
                trees[i],
                0,
                trees[grp[0]],
                0,
                keyfunc=lambda n: f'{n["element"]}{n["formalcharge"]}',
            ):
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
    methyl.add_node(0, element="C", formalcharge=0)
    methyl.add_node(1, element="H", formalcharge=0)
    methyl.add_node(2, element="H", formalcharge=0)
    methyl.add_node(3, element="H", formalcharge=0)
    methyl.add_edge(0, 1)
    methyl.add_edge(0, 2)
    methyl.add_edge(0, 3)

    return methyl


def _getHydroxylGraph():
    """
    Generate a molecular graph for hydroxyl group
    """

    gg = nx.Graph()
    gg.add_node(0, element="O", formalcharge=0)
    gg.add_node(1, element="H", formalcharge=0)
    gg.add_edge(0, 1)

    return gg


def connected_component_subgraphs(graph):
    return (
        graph.subgraph(component).copy() for component in nx.connected_components(graph)
    )


def detectParameterizableCores(
    graph, skip_methyl=True, skip_hydroxyl=False, skip_terminal_hs=False
):
    """
    Detect parametrizable dihedral angle cores (central atom pairs)

    The cores are detected by looking for bridges (bonds which divide the molecule into two parts) in a molecular graph.
    Terminal cores are skipped.
    """

    methyl = _getMethylGraph()
    hydroxyl = _getHydroxylGraph()

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
        if skip_methyl and (
            _checkIsomorphism(sideGraphs[0], methyl)
            or _checkIsomorphism(sideGraphs[1], methyl)
        ):
            continue

        # Skip if a side graph is a hydroxyl group
        if skip_hydroxyl and (
            _checkIsomorphism(sideGraphs[0], hydroxyl)
            or _checkIsomorphism(sideGraphs[1], hydroxyl)
        ):
            continue

        # Skip if core contains C with sp hybridization
        if graph.nodes[core[0]]["element"] == "C" and graph.degree(core[0]) == 2:
            continue
        if graph.nodes[core[1]]["element"] == "C" and graph.degree(core[1]) == 2:
            continue

        # Skip if side contains only Hs
        if skip_terminal_hs:
            keys_0 = [x for x in sideGraphs[0].nodes if x != core[0]]
            keys_1 = [x for x in sideGraphs[1].nodes if x != core[1]]
            if all([graph.nodes[i]["element"] == "H" for i in keys_0]):
                continue
            if all([graph.nodes[i]["element"] == "H" for i in keys_1]):
                continue

        all_core_sides.append((core, sideGraphs))

    return all_core_sides


def _weighted_closeness_centrality(graph, node, weightfunc=None):
    """
    Weighted closeness centrality

    Identical to networkx.closeness_centrality, except the shorted path lengths are weighted
    by a function which takes as input the node and outputs a score.
    """

    lengths = nx.shortest_path_length(graph, source=node)
    del lengths[node]
    weights = (
        {node_: weightfunc(graph.nodes[node_]) for node_ in lengths}
        if weightfunc
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

    def _weightfunc(node):
        # Give following score to each atom:
        # element number + formal charge * 0.1 + 0.5
        # The reasoning is that with formal charge 0 you will get for C 6.5
        # With -1 formal charge you will get 6.49, with +1 formal charge you will get 6.56
        # That way we can differentiate between formal charges and elements
        return node["number"] + node["formalcharge"] * 0.01 + 0.5

    weightedCentralities = [
        _weighted_closeness_centrality(graph, terminal, weightfunc=_weightfunc)
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
            logger.warning(
                "Molecular scoring function is not sufficient. "
                "Dihedral selection depends on the atom order! "
                "Redundant dihedrals might be present!"
            )

    return chosen_terminals


def detectParameterizableDihedrals(
    molecule,
    exclude_atoms=(),
    return_all_dihedrals=False,
    skip_methyl=True,
    skip_hydroxyl=False,
    skip_terminal_hs=False,
):
    """
    Detect parameterizable dihedral angles

    Arguments
    ---------
    molecule : :class:`Molecule <moleculekit.molecule.Molecule>`
        Molecule object
    exclude_atoms : list
        Ignore dihedrals which consist purely of atoms in this list
    return_all_dihedrals : bool
        Return all dihedral terms. When False it filters out and selects only the dihedral
        with the terminal with the highest centrality for each core.
    skip_methyl : bool
        Setting to True will skip dihedrals whose terminal is a methyl group
    skip_hydroxyl : bool
        Setting to True will skip dihedrals whose terminal is a hydroxyl group
    skip_terminal_hs : bool
        Setting to True will skip dihedrals ending in hydrogens

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
    >>> molFile = 'glycol.mol2'
    >>> mol = Molecule(molFile, guess=('bonds', 'angles', 'dihedrals'))
    >>> detectParameterizableDihedrals(mol)
    [[(0, 1, 2, 3)], [(1, 2, 3, 9), (2, 1, 0, 4)]]

    Find the parameterizable dihedrals of ethanolamine
    >>> molFile = 'ethanolamine.mol2'
    >>> mol = Molecule(molFile, guess=('bonds', 'angles', 'dihedrals'))
    >>> detectParameterizableDihedrals(mol)
    [[(0, 1, 2, 3)], [(1, 2, 3, 9), (1, 2, 3, 10)], [(2, 1, 0, 4)]]

    Find the parameterizable dihedrals of benzamidine
    >>> molFile = 'benzamidine.mol2'
    >>> mol = Molecule(molFile, guess=('bonds', 'angles', 'dihedrals'))
    >>> detectParameterizableDihedrals(mol)
    [[(0, 6, 12, 16), (0, 6, 12, 17), (0, 6, 13, 14), (0, 6, 13, 15)], [(1, 0, 6, 12), (1, 0, 6, 13), (5, 0, 6, 12), (5, 0, 6, 13)]]

    # Check if the atom swapping does not affect results

    Find the parameterizable dihedrals of chlorethene
    >>> molFile = 'chlorethene_1.mol2'
    >>> mol = Molecule(molFile, guess=('bonds', 'angles', 'dihedrals'))
    >>> detectParameterizableDihedrals(mol)
    [[(2, 1, 0, 4), (2, 1, 0, 5)]]

    Find the parameterizable dihedrals of chlorethene (with swapped atoms)
    >>> molFile = 'chlorethene_2.mol2'
    >>> mol = Molecule(molFile, guess=('bonds', 'angles', 'dihedrals'))
    >>> detectParameterizableDihedrals(mol)
    [[(3, 1, 0, 4), (3, 1, 0, 5)]]

    # Check if triple bonds are skipped

    Find the parameterizable dihedrals of 4-hexinenitrile
    >>> molFile = '4-hexinenitrile.mol2'
    >>> mol = Molecule(molFile, guess=('bonds', 'angles', 'dihedrals'))
    >>> detectParameterizableDihedrals(mol)
    [[(2, 3, 4, 5)]]

    # Check the scoring function

    Find the parameterizable dihedrals of dicarbothioic acid
    >>> molFile = 'dicarbothioic_acid.mol2'
    >>> mol = Molecule(molFile, guess=('bonds', 'angles', 'dihedrals'))
    >>> detectParameterizableDihedrals(mol)
    [[(0, 1, 3, 5)], [(1, 3, 5, 7)], [(3, 1, 0, 6)]]

    Find the parameterizable dihedrals of 2-hydroxypyridine
    >>> molFile = '2-hydroxypyridine.mol2'
    >>> mol = Molecule(molFile, guess=('bonds', 'angles', 'dihedrals'))
    >>> detectParameterizableDihedrals(mol)
    [[(6, 1, 0, 7)]]

    Find the parameterizable dihedrals of fluorchlorcyclopronol
    >>> molFile = 'fluorchlorcyclopronol.mol2'
    >>> mol = Molecule(molFile, guess=('bonds', 'angles', 'dihedrals'))
    >>> detectParameterizableDihedrals(mol)
    [[(2, 4, 5, 9)]]
    """
    # Get a molecular graph
    graph = _getMolecularGraph(molecule)

    # Get the equivalence groups for each atom in the molecule
    _, _, equivalent_group_by_atom = detectEquivalentAtoms(molecule)

    # Get parameterizable dihedral angles
    chosen_groups = []
    all_dihedrals = []
    for core, sides in detectParameterizableCores(
        graph,
        skip_methyl=skip_methyl,
        skip_hydroxyl=skip_hydroxyl,
        skip_terminal_hs=skip_terminal_hs,
    ):
        # Keep all terminals
        all_terminals = [
            list(side.neighbors(centre)) for centre, side in zip(core, sides)
        ]
        # Choose the best terminals for each side
        chosen_left = _chooseTerminals(graph, core[0], sides[0])
        chosen_right = _chooseTerminals(graph, core[1], sides[1])

        # Generate all terminal combinations for all dihedrals
        for terminals in itertools.product(*all_terminals):
            new_dihedral = (terminals[0], *core, terminals[1])
            if all([idx in exclude_atoms for idx in new_dihedral]):
                # Skip dihedrals consisting purely of atoms in exclude_atoms
                continue
            all_dihedrals.append(new_dihedral)

            # Add dihedral group to chosen if the terminals match the chosen ones
            groups = [equivalent_group_by_atom[x] for x in new_dihedral]
            groups = groups[::-1] if groups[::-1] < groups else groups
            if terminals[0] in chosen_left and terminals[1] in chosen_right:
                chosen_groups.append(tuple(groups))

    # Get equivalent groups for each atom for each dihedral
    dihedral_groups = [
        tuple([equivalent_group_by_atom[atom] for atom in dihedral])
        for dihedral in all_dihedrals
    ]

    # Group equivalent dihedral angles and reverse them that equivalent atoms are matched
    equivalent_dihedrals = {}
    for dihedral, groups in zip(all_dihedrals, dihedral_groups):
        dihedral, groups = (
            (dihedral[::-1], groups[::-1])
            if groups[::-1] < groups
            else (dihedral, groups)
        )
        equivalent_dihedrals[groups] = sorted(
            equivalent_dihedrals.get(groups, []) + [dihedral]
        )

    # Keep only chosen dihedrals and filter out multiple dihedrals per dihedral bond
    if not return_all_dihedrals:
        from collections import defaultdict

        # Delete all groups which were not chosen
        to_delete = [
            group for group in equivalent_dihedrals if group not in chosen_groups
        ]
        for td in to_delete:
            del equivalent_dihedrals[td]

        dihedral_dict = defaultdict(list)
        for dih in equivalent_dihedrals.values():
            dihedral_dict[tuple(sorted(dih[0][1:3]))].append(dih)

        equivalent_dihedrals = []
        for core in dihedral_dict:
            equiv_lens = [len(x) for x in dihedral_dict[core]]
            max_idx = equiv_lens.index(max(equiv_lens))
            equivalent_dihedrals.append(dihedral_dict[core][max_idx])
    else:
        equivalent_dihedrals = equivalent_dihedrals.values()

    return sorted(equivalent_dihedrals)
