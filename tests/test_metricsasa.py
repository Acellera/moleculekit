from moleculekit.projections.metricsasa import MetricSasa
from moleculekit.molecule import Molecule
import numpy as np
import pytest
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def _mol():
    mol = Molecule(
        os.path.join(curr_dir, "test_projections", "trajectory", "filtered.pdb")
    )
    mol.read(os.path.join(curr_dir, "test_projections", "trajectory", "traj.xtc"))
    mol.dropFrames(keep=[0, 1])  # Keep only two frames because it's super slow
    return mol


def _test_sasa_atom(_mol):
    metr = MetricSasa(mode="atom")
    sasaA = metr.project(_mol.copy())
    sasaA_ref = np.load(
        os.path.join(curr_dir, "test_projections", "metricsasa", "sasa_atom.npy")
    )
    assert np.allclose(
        sasaA, sasaA_ref, atol=0.1
    ), f"Failed with max diff {np.abs(sasaA-sasaA_ref).max()}"


def _test_sasa_residue(_mol):
    metr = MetricSasa(mode="residue")
    sasaR = metr.project(_mol.copy())
    sasaR_ref = np.load(
        os.path.join(curr_dir, "test_projections", "metricsasa", "sasa_residue.npy")
    )
    assert np.allclose(
        sasaR, sasaR_ref, atol=0.3
    ), f"Failed with max diff {np.abs(sasaR-sasaR_ref).max()}"


def _test_set_diff_error(_mol):
    try:
        metr = MetricSasa(mode="atom", sel="index 3000", filtersel="not index 3000")
        metr._calculateMolProp(_mol.copy())
    except RuntimeError:
        print("Correctly threw a runtime error for bad selections")
    else:
        raise AssertionError(
            "This should throw an error as the selected atom does not exist in the filtered system"
        )


def _test_selection_and_filtering(_mol):
    metr = MetricSasa(mode="atom")
    sasaA_ref = metr.project(_mol.copy())

    metr = MetricSasa(mode="atom", sel="index 20")  # Get just the SASA of the 20th atom
    sasaA = metr.project(_mol.copy())
    assert np.allclose(
        sasaA, sasaA_ref[:, [20]], atol=1e-2
    ), f"SASA atom selection failed to give same results as without selection. Max diff: {np.abs(sasaA-sasaA_ref[:, [20]]).max()}"

    metr = MetricSasa(
        mode="atom", sel="index 20", filtersel="index 20"
    )  # Get just the SASA of the 20th atom, remove all else
    sasaA = metr.project(_mol.copy())
    assert not np.allclose(
        sasaA, sasaA_ref[:, [20]], atol=1e-2
    ), "SASA filtering gave same results as without filtering. Bad."


def _test_mappings(_mol):
    metr = MetricSasa(mode="atom")
    mapping = metr.getMapping(_mol)
    assert np.array_equal(mapping.atomIndexes, np.arange(4480))

    # fmt: off
    metr = MetricSasa(mode='residue')
    mapping = metr.getMapping(_mol)
    ref = np.array([0,   17,   27,   39,   49,   56,   75,   99,  113,  132,  152, 167,  189,  211,  222,  241,  256,  268,  290,  304,  319,  343,
                    358,  377,  396,  411,  422,  443,  463,  484,  500,  515,  522, 533,  545,  555,  570,  589,  596,  613,  624,  638,  662,  679,
                    695,  712,  731,  751,  775,  797,  808,  822,  839,  854,  873, 892,  902,  909,  919,  930,  949,  968,  979,  991, 1015, 1039,
                    1055, 1074, 1088, 1098, 1108, 1125, 1135, 1154, 1173, 1194, 1208, 1222, 1246, 1258, 1280, 1294, 1314, 1328, 1343, 1357, 1369, 1388,
                    1407, 1423, 1447, 1466, 1473, 1495, 1512, 1523, 1547, 1561, 1585, 1606, 1621, 1645, 1659, 1678, 1693, 1715, 1734, 1745, 1762, 1781,
                    1796, 1818, 1837, 1858, 1877, 1894, 1908, 1932, 1953, 1967, 1991, 2015, 2030, 2044, 2063, 2075, 2099, 2111, 2130, 2140, 2159, 2176,
                    2198, 2217, 2239, 2261, 2275, 2291, 2301, 2321, 2332, 2344, 2365, 2384, 2401, 2415, 2431, 2441, 2460, 2474, 2486, 2510, 2525, 2539,
                    2549, 2559, 2570, 2589, 2608, 2625, 2635, 2642, 2663, 2685, 2692, 2716, 2732, 2746, 2753, 2777, 2784, 2798, 2817, 2839, 2854, 2861,
                    2878, 2892, 2903, 2919, 2938, 2955, 2971, 2987, 3001, 3020, 3034, 3053, 3069, 3084, 3108, 3122, 3138, 3148, 3170, 3182, 3193, 3207,
                    3231, 3250, 3274, 3293, 3307, 3319, 3333, 3350, 3370, 3380, 3390, 3397, 3418, 3440, 3454, 3466, 3481, 3488, 3510, 3534, 3541, 3553,
                    3563, 3573, 3588, 3595, 3607, 3618, 3625, 3632, 3646, 3666, 3682, 3699, 3721, 3732, 3746, 3766, 3780, 3794, 3818, 3842, 3863, 3880,
                    3897, 3904, 3923, 3939, 3950, 3974, 3981, 3996, 4003, 4013, 4025, 4049, 4061, 4068, 4090, 4111, 4118, 4138, 4159, 4173, 4190, 4206,
                    4226, 4250, 4269, 4291, 4313, 4337, 4356, 4373, 4395, 4411, 4430, 4442, 4459])
    # fmt: on
    assert np.array_equal(mapping.atomIndexes, ref)
