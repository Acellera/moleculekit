import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


def _compare(arr1, arr2):
    import numpy as np

    assert np.array_equal(
        np.array(arr1, dtype=object), np.array(arr2, dtype=object)
    ), f"Different results:\n{arr1}\n{arr2}"


def _test_atom_detection():
    import os
    from moleculekit.molecule import Molecule
    from moleculekit.tools.detect import detectEquivalentAtoms

    mol = Molecule(os.path.join(curr_dir, "test_detect", "KCX.cif"))
    eqgroups, eqatoms, eqgroupbyatom = detectEquivalentAtoms(mol)

    # fmt: off
    eqgroups_ref = [(0, 2, 3), (1,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11, 12, 13), (14,), (15,), (16,), (17,), (18,), (19,), (20,), (21,), (22,), (23,), (24,), (25,), (26,), (27,), (28, 29), (30, 31), (32, 33), (34, 35), (36,), (37,), (38,), (39,), (40,), (41,), (42,), (43,), (44, 45, 46), (47,), (48,), (49,), (50,), (51,), (52, 53, 54)]
    eqatoms_ref = [(0, 2, 3), (1,), (0, 2, 3), (0, 2, 3), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11, 12, 13), (11, 12, 13), (11, 12, 13), (14,), (15,), (16,), (17,), (18,), (19,), (20,), (21,), (22,), (23,), (24,), (25,), (26,), (27,), (28, 29), (28, 29), (30, 31), (30, 31), (32, 33), (32, 33), (34, 35), (34, 35), (36,), (37,), (38,), (39,), (40,), (41,), (42,), (43,), (44, 45, 46), (44, 45, 46), (44, 45, 46), (47,), (48,), (49,), (50,), (51,), (52, 53, 54), (52, 53, 54), (52, 53, 54)]
    eqgroupbyatom_ref = [0, 1, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 36, 36, 37, 38, 39, 40, 41, 42, 42, 42]
    # fmt: on
    _compare(eqgroups, eqgroups_ref)
    _compare(eqatoms, eqatoms_ref)
    _compare(eqgroupbyatom, eqgroupbyatom_ref)


def _test_charged_detection():
    import os
    from moleculekit.molecule import Molecule
    from moleculekit.tools.detect import detectEquivalentAtoms

    mol = Molecule(os.path.join(curr_dir, "test_detect", "BEN_pH7.4.sdf"))
    eqgroups, eqatoms, eqgroupbyatom = detectEquivalentAtoms(mol)

    # fmt: off
    eqgroups_ref = [(0,), (1, 5), (2, 4), (3,), (6,), (7,), (8,), (9, 13), (10, 12), (11,), (14, 15), (16, 17)]
    eqatoms_ref = [(0,), (1, 5), (2, 4), (3,), (2, 4), (1, 5), (6,), (7,), (8,), (9, 13), (10, 12), (11,), (10, 12), (9, 13), (14, 15), (14, 15), (16, 17), (16, 17)]
    eqgroupbyatom_ref = [0, 1, 2, 3, 2, 1, 4, 5, 6, 7, 8, 9, 8, 7, 10, 10, 11, 11]
    # fmt: on
    _compare(eqgroups, eqgroups_ref)
    _compare(eqatoms, eqatoms_ref)
    _compare(eqgroupbyatom, eqgroupbyatom_ref)

    # Set formal charge to 0 and try again
    mol.formalcharge[8] = 0
    eqgroups, eqatoms, eqgroupbyatom = detectEquivalentAtoms(mol)
    # fmt: off
    eqgroups_ref = [(0,), (1, 5), (2, 4), (3,), (6,), (7, 8), (9, 13), (10, 12), (11,), (14, 15, 16, 17)]
    eqatoms_ref = [(0,), (1, 5), (2, 4), (3,), (2, 4), (1, 5), (6,), (7, 8), (7, 8), (9, 13), (10, 12), (11,), (10, 12), (9, 13), (14, 15, 16, 17), (14, 15, 16, 17), (14, 15, 16, 17), (14, 15, 16, 17)]
    eqgroupbyatom_ref = [0, 1, 2, 3, 2, 1, 4, 5, 5, 6, 7, 8, 7, 6, 9, 9, 9, 9]
    # fmt: on
    _compare(eqgroups, eqgroups_ref)
    _compare(eqatoms, eqatoms_ref)
    _compare(eqgroupbyatom, eqgroupbyatom_ref)


def _test_detect_dihedrals():
    import os
    from moleculekit.molecule import Molecule
    from moleculekit.tools.detect import detectParameterizableDihedrals

    mol = Molecule(os.path.join(curr_dir, "test_detect", "m19.cif"))
    res = detectParameterizableDihedrals(mol, skip_terminal_hs=False)
    ref = [
        [(11, 9, 8, 12)],
        [(12, 13, 14, 15)],
        [(13, 14, 15, 17)],
        [(14, 13, 12, 18)],
        [(18, 6, 5, 20)],
        [(20, 1, 0, 21), (20, 1, 0, 22)],
    ]
    assert res == ref, f"{res}, {ref}"

    res = detectParameterizableDihedrals(mol, skip_terminal_hs=True)
    ref = [
        [(11, 9, 8, 12)],
        [(12, 13, 14, 15)],
        [(13, 14, 15, 17)],
        [(14, 13, 12, 18)],
        [(18, 6, 5, 20)],
    ]
    assert res == ref, f"{res}, {ref}"

    res = detectParameterizableDihedrals(
        mol, skip_terminal_hs=True, return_all_dihedrals=True
    )
    ref = [
        [(4, 5, 6, 7)],
        [(4, 5, 6, 18)],
        [(7, 6, 5, 20)],
        [(7, 8, 9, 10)],
        [(7, 8, 9, 11)],
        [(8, 12, 13, 14)],
        [(10, 9, 8, 12)],
        [(11, 9, 8, 12)],
        [(12, 13, 14, 15)],
        [(12, 13, 14, 26), (12, 13, 14, 27)],
        [(13, 14, 15, 16)],
        [(13, 14, 15, 17)],
        [(14, 13, 12, 18)],
        [(16, 15, 14, 26), (16, 15, 14, 27)],
        [(17, 15, 14, 26), (17, 15, 14, 27)],
        [(18, 6, 5, 20)],
    ]
    assert res == ref, f"{res}\n{ref}"

    # Interesting case because if you don't use formal charges in the weighted centrality
    # score it messes up by choosing once the charged (in one of the copies)
    # and once the uncharged atom (in the other 3 copies of the terminal)
    mol = Molecule(os.path.join(curr_dir, "test_detect", "OAH_Sampl5_host.cif"))
    res = detectParameterizableDihedrals(mol, skip_terminal_hs=True)
    ref = [
        [
            (0, 31, 72, 75),
            (7, 9, 64, 67),
            (15, 9, 64, 67),
            (20, 23, 56, 59),
            (25, 23, 56, 59),
            (33, 31, 72, 75),
            (48, 51, 80, 83),
            (53, 51, 80, 83),
        ],
        [(9, 64, 67, 70), (23, 56, 59, 62), (31, 72, 75, 78), (51, 80, 83, 86)],
        [(56, 59, 62, 63), (64, 67, 70, 71), (72, 75, 78, 79), (80, 83, 86, 166)],
        [
            (132, 138, 168, 172),
            (136, 138, 168, 172),
            (142, 149, 163, 173),
            (144, 149, 163, 173),
            (151, 158, 159, 171),
            (153, 158, 159, 171),
            (176, 182, 161, 170),
            (180, 182, 161, 170),
        ],
    ]
    assert res == ref
