# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
from collections import namedtuple
import logging

logger = logging.getLogger(__file__)

# from periodictable import elements  # the python package called periodictable, not this file
# for el in elements:
#     if el.name == 'neutron':
#         continue
#     # if mendeleev.__dict__[el.symbol].vdw_radius_bondi is not None:
#     #     vdw_radius = mendeleev.__dict__[el.symbol].vdw_radius_bondi
#     # else:
#     vdw_radius = mendeleev.__dict__[el.symbol].vdw_radius
#     if vdw_radius is not None:
#         vdw_radius /= 100  # Convert picometer to A
#     print("'{}': _Element(symbol='{}', name='{}', number={}, mass={}, vdw_radius={}),".format(el, el, el.name, el.number, el.mass, vdw_radius))


_Element = namedtuple("Element", ["symbol", "name", "number", "mass", "vdw_radius"])
periodictable = {
    "H": _Element(symbol="H", name="hydrogen", number=1, mass=1.00794, vdw_radius=1.1),
    "He": _Element(symbol="He", name="helium", number=2, mass=4.002602, vdw_radius=1.4),
    "Li": _Element(symbol="Li", name="lithium", number=3, mass=6.941, vdw_radius=1.82),
    "Be": _Element(
        symbol="Be", name="beryllium", number=4, mass=9.012182, vdw_radius=1.53
    ),
    "B": _Element(symbol="B", name="boron", number=5, mass=10.811, vdw_radius=1.92),
    "C": _Element(symbol="C", name="carbon", number=6, mass=12.0107, vdw_radius=1.7),
    "N": _Element(symbol="N", name="nitrogen", number=7, mass=14.0067, vdw_radius=1.55),
    "O": _Element(symbol="O", name="oxygen", number=8, mass=15.9994, vdw_radius=1.52),
    "F": _Element(
        symbol="F", name="fluorine", number=9, mass=18.9984032, vdw_radius=1.47
    ),
    "Ne": _Element(symbol="Ne", name="neon", number=10, mass=20.1797, vdw_radius=1.54),
    "Na": _Element(
        symbol="Na", name="sodium", number=11, mass=22.98977, vdw_radius=2.27
    ),
    "Mg": _Element(
        symbol="Mg", name="magnesium", number=12, mass=24.305, vdw_radius=1.73
    ),
    "Al": _Element(
        symbol="Al", name="aluminum", number=13, mass=26.981538, vdw_radius=1.84
    ),
    "Si": _Element(
        symbol="Si", name="silicon", number=14, mass=28.0855, vdw_radius=2.1
    ),
    "P": _Element(
        symbol="P", name="phosphorus", number=15, mass=30.973761, vdw_radius=1.8
    ),
    "S": _Element(symbol="S", name="sulfur", number=16, mass=32.065, vdw_radius=1.8),
    "Cl": _Element(
        symbol="Cl", name="chlorine", number=17, mass=35.453, vdw_radius=1.75
    ),
    "Ar": _Element(symbol="Ar", name="argon", number=18, mass=39.948, vdw_radius=1.88),
    "K": _Element(
        symbol="K", name="potassium", number=19, mass=39.0983, vdw_radius=2.75
    ),
    "Ca": _Element(
        symbol="Ca", name="calcium", number=20, mass=40.078, vdw_radius=2.31
    ),
    "Sc": _Element(
        symbol="Sc", name="scandium", number=21, mass=44.95591, vdw_radius=2.15
    ),
    "Ti": _Element(
        symbol="Ti", name="titanium", number=22, mass=47.867, vdw_radius=2.11
    ),
    "V": _Element(
        symbol="V", name="vanadium", number=23, mass=50.9415, vdw_radius=2.07
    ),
    "Cr": _Element(
        symbol="Cr", name="chromium", number=24, mass=51.9961, vdw_radius=2.06
    ),
    "Mn": _Element(
        symbol="Mn", name="manganese", number=25, mass=54.938049, vdw_radius=2.05
    ),
    "Fe": _Element(symbol="Fe", name="iron", number=26, mass=55.845, vdw_radius=2.04),
    "Co": _Element(symbol="Co", name="cobalt", number=27, mass=58.9332, vdw_radius=2.0),
    "Ni": _Element(
        symbol="Ni", name="nickel", number=28, mass=58.6934, vdw_radius=1.97
    ),
    "Cu": _Element(symbol="Cu", name="copper", number=29, mass=63.546, vdw_radius=1.96),
    "Zn": _Element(symbol="Zn", name="zinc", number=30, mass=65.409, vdw_radius=2.01),
    "Ga": _Element(
        symbol="Ga", name="gallium", number=31, mass=69.723, vdw_radius=1.87
    ),
    "Ge": _Element(
        symbol="Ge", name="germanium", number=32, mass=72.64, vdw_radius=2.11
    ),
    "As": _Element(
        symbol="As", name="arsenic", number=33, mass=74.9216, vdw_radius=1.85
    ),
    "Se": _Element(symbol="Se", name="selenium", number=34, mass=78.96, vdw_radius=1.9),
    "Br": _Element(
        symbol="Br", name="bromine", number=35, mass=79.904, vdw_radius=1.85
    ),
    "Kr": _Element(
        symbol="Kr", name="krypton", number=36, mass=83.798, vdw_radius=2.02
    ),
    "Rb": _Element(
        symbol="Rb", name="rubidium", number=37, mass=85.4678, vdw_radius=3.03
    ),
    "Sr": _Element(
        symbol="Sr", name="strontium", number=38, mass=87.62, vdw_radius=2.49
    ),
    "Y": _Element(
        symbol="Y", name="yttrium", number=39, mass=88.90585, vdw_radius=2.32
    ),
    "Zr": _Element(
        symbol="Zr", name="zirconium", number=40, mass=91.224, vdw_radius=2.23
    ),
    "Nb": _Element(
        symbol="Nb", name="niobium", number=41, mass=92.90638, vdw_radius=2.18
    ),
    "Mo": _Element(
        symbol="Mo", name="molybdenum", number=42, mass=95.94, vdw_radius=2.17
    ),
    "Tc": _Element(symbol="Tc", name="technetium", number=43, mass=98, vdw_radius=2.16),
    "Ru": _Element(
        symbol="Ru", name="ruthenium", number=44, mass=101.07, vdw_radius=2.13
    ),
    "Rh": _Element(
        symbol="Rh", name="rhodium", number=45, mass=102.9055, vdw_radius=2.1
    ),
    "Pd": _Element(
        symbol="Pd", name="palladium", number=46, mass=106.42, vdw_radius=2.1
    ),
    "Ag": _Element(
        symbol="Ag", name="silver", number=47, mass=107.8682, vdw_radius=2.11
    ),
    "Cd": _Element(
        symbol="Cd", name="cadmium", number=48, mass=112.411, vdw_radius=2.18
    ),
    "In": _Element(
        symbol="In", name="indium", number=49, mass=114.818, vdw_radius=1.93
    ),
    "Sn": _Element(symbol="Sn", name="tin", number=50, mass=118.71, vdw_radius=2.17),
    "Sb": _Element(
        symbol="Sb", name="antimony", number=51, mass=121.76, vdw_radius=2.06
    ),
    "Te": _Element(
        symbol="Te", name="tellurium", number=52, mass=127.6, vdw_radius=2.06
    ),
    "I": _Element(
        symbol="I", name="iodine", number=53, mass=126.90447, vdw_radius=1.98
    ),
    "Xe": _Element(symbol="Xe", name="xenon", number=54, mass=131.293, vdw_radius=2.16),
    "Cs": _Element(
        symbol="Cs", name="cesium", number=55, mass=132.90545, vdw_radius=3.43
    ),
    "Ba": _Element(
        symbol="Ba", name="barium", number=56, mass=137.327, vdw_radius=2.68
    ),
    "La": _Element(
        symbol="La", name="lanthanum", number=57, mass=138.9055, vdw_radius=2.43
    ),
    "Ce": _Element(
        symbol="Ce", name="cerium", number=58, mass=140.116, vdw_radius=2.42
    ),
    "Pr": _Element(
        symbol="Pr", name="praseodymium", number=59, mass=140.90765, vdw_radius=2.4
    ),
    "Nd": _Element(
        symbol="Nd", name="neodymium", number=60, mass=144.24, vdw_radius=2.39
    ),
    "Pm": _Element(
        symbol="Pm", name="promethium", number=61, mass=145, vdw_radius=2.38
    ),
    "Sm": _Element(
        symbol="Sm", name="samarium", number=62, mass=150.36, vdw_radius=2.36
    ),
    "Eu": _Element(
        symbol="Eu", name="europium", number=63, mass=151.964, vdw_radius=2.35
    ),
    "Gd": _Element(
        symbol="Gd", name="gadolinium", number=64, mass=157.25, vdw_radius=2.34
    ),
    "Tb": _Element(
        symbol="Tb", name="terbium", number=65, mass=158.92534, vdw_radius=2.33
    ),
    "Dy": _Element(
        symbol="Dy", name="dysprosium", number=66, mass=162.5, vdw_radius=2.31
    ),
    "Ho": _Element(
        symbol="Ho", name="holmium", number=67, mass=164.93032, vdw_radius=2.3
    ),
    "Er": _Element(
        symbol="Er", name="erbium", number=68, mass=167.259, vdw_radius=2.29
    ),
    "Tm": _Element(
        symbol="Tm", name="thulium", number=69, mass=168.93421, vdw_radius=2.27
    ),
    "Yb": _Element(
        symbol="Yb", name="ytterbium", number=70, mass=173.04, vdw_radius=2.26
    ),
    "Lu": _Element(
        symbol="Lu", name="lutetium", number=71, mass=174.967, vdw_radius=2.24
    ),
    "Hf": _Element(
        symbol="Hf", name="hafnium", number=72, mass=178.49, vdw_radius=2.23
    ),
    "Ta": _Element(
        symbol="Ta", name="tantalum", number=73, mass=180.9479, vdw_radius=2.22
    ),
    "W": _Element(symbol="W", name="tungsten", number=74, mass=183.84, vdw_radius=2.18),
    "Re": _Element(
        symbol="Re", name="rhenium", number=75, mass=186.207, vdw_radius=2.16
    ),
    "Os": _Element(symbol="Os", name="osmium", number=76, mass=190.23, vdw_radius=2.16),
    "Ir": _Element(
        symbol="Ir", name="iridium", number=77, mass=192.217, vdw_radius=2.13
    ),
    "Pt": _Element(
        symbol="Pt", name="platinum", number=78, mass=195.078, vdw_radius=2.13
    ),
    "Au": _Element(
        symbol="Au", name="gold", number=79, mass=196.96655, vdw_radius=2.14
    ),
    "Hg": _Element(
        symbol="Hg", name="mercury", number=80, mass=200.59, vdw_radius=2.23
    ),
    "Tl": _Element(
        symbol="Tl", name="thallium", number=81, mass=204.3833, vdw_radius=1.96
    ),
    "Pb": _Element(symbol="Pb", name="lead", number=82, mass=207.2, vdw_radius=2.02),
    "Bi": _Element(
        symbol="Bi", name="bismuth", number=83, mass=208.98038, vdw_radius=2.07
    ),
    "Po": _Element(symbol="Po", name="polonium", number=84, mass=209, vdw_radius=1.97),
    "At": _Element(symbol="At", name="astatine", number=85, mass=210, vdw_radius=2.02),
    "Rn": _Element(symbol="Rn", name="radon", number=86, mass=222, vdw_radius=2.2),
    "Fr": _Element(symbol="Fr", name="francium", number=87, mass=223, vdw_radius=3.48),
    "Ra": _Element(symbol="Ra", name="radium", number=88, mass=226, vdw_radius=2.83),
    "Ac": _Element(symbol="Ac", name="actinium", number=89, mass=227, vdw_radius=2.47),
    "Th": _Element(
        symbol="Th", name="thorium", number=90, mass=232.0381, vdw_radius=2.45
    ),
    "Pa": _Element(
        symbol="Pa", name="protactinium", number=91, mass=231.03588, vdw_radius=2.43
    ),
    "U": _Element(
        symbol="U", name="uranium", number=92, mass=238.02891, vdw_radius=2.41
    ),
    "Np": _Element(symbol="Np", name="neptunium", number=93, mass=237, vdw_radius=2.39),
    "Pu": _Element(symbol="Pu", name="plutonium", number=94, mass=244, vdw_radius=2.43),
    "Am": _Element(symbol="Am", name="americium", number=95, mass=243, vdw_radius=2.44),
    "Cm": _Element(symbol="Cm", name="curium", number=96, mass=247, vdw_radius=2.45),
    "Bk": _Element(symbol="Bk", name="berkelium", number=97, mass=247, vdw_radius=2.44),
    "Cf": _Element(
        symbol="Cf", name="californium", number=98, mass=251, vdw_radius=2.45
    ),
    "Es": _Element(
        symbol="Es", name="einsteinium", number=99, mass=252, vdw_radius=2.45
    ),
    "Fm": _Element(symbol="Fm", name="fermium", number=100, mass=257, vdw_radius=2.45),
    "Md": _Element(
        symbol="Md", name="mendelevium", number=101, mass=258, vdw_radius=2.46
    ),
    "No": _Element(symbol="No", name="nobelium", number=102, mass=259, vdw_radius=2.46),
    "Lr": _Element(
        symbol="Lr", name="lawrencium", number=103, mass=262, vdw_radius=2.46
    ),
    "Rf": _Element(
        symbol="Rf", name="rutherfordium", number=104, mass=261, vdw_radius=None
    ),
    "Db": _Element(symbol="Db", name="dubnium", number=105, mass=262, vdw_radius=None),
    "Sg": _Element(
        symbol="Sg", name="seaborgium", number=106, mass=266, vdw_radius=None
    ),
    "Bh": _Element(symbol="Bh", name="bohrium", number=107, mass=264, vdw_radius=None),
    "Hs": _Element(symbol="Hs", name="hassium", number=108, mass=277, vdw_radius=None),
    "Mt": _Element(
        symbol="Mt", name="meitnerium", number=109, mass=268, vdw_radius=None
    ),
    "Ds": _Element(
        symbol="Ds", name="darmstadtium", number=110, mass=281, vdw_radius=None
    ),
    "Rg": _Element(
        symbol="Rg", name="roentgenium", number=111, mass=272, vdw_radius=None
    ),
    "Cn": _Element(
        symbol="Cn", name="copernicium", number=112, mass=285, vdw_radius=None
    ),
    "Nh": _Element(symbol="Nh", name="nihonium", number=113, mass=286, vdw_radius=None),
    "Fl": _Element(
        symbol="Fl", name="flerovium", number=114, mass=289, vdw_radius=None
    ),
    "Mc": _Element(
        symbol="Mc", name="moscovium", number=115, mass=289, vdw_radius=None
    ),
    "Lv": _Element(
        symbol="Lv", name="livermorium", number=116, mass=293, vdw_radius=None
    ),
    "Ts": _Element(
        symbol="Ts", name="tennessine", number=117, mass=294, vdw_radius=None
    ),
    "Og": _Element(
        symbol="Og", name="oganesson", number=118, mass=294, vdw_radius=None
    ),
}

# Add indexes as well by atomic number
periodictable_by_number = {
    periodictable[el].number: val for el, val in periodictable.items()
}

import numpy as np

# This of course fails for exotic elements like Bk-Cm Db-Lr Mc-Fl Og-Ts which have similar masses
_all_elements = np.array([el for el in periodictable])
_all_masses = np.array([periodictable[el].mass for el in periodictable])


def elements_from_masses(masses):
    from moleculekit.util import ensurelist
    from scipy.spatial.distance import cdist

    masses = np.array(ensurelist(masses))
    if np.any(masses > 140):
        logger.warning(
            "Guessing element for atoms with mass > 140. This can lead to inaccurate element guesses."
        )
    elements = _all_elements[
        np.argmin(cdist(masses[:, None], np.array(_all_masses)[:, None]), axis=1)
    ]

    # Fix for zero masses. Should not guess hydrogens.
    elements[masses == 0] = ""

    elements = elements.tolist()
    if len(elements) == 1:
        return elements[0]
    return elements


import unittest


class _TestPeriodicTable(unittest.TestCase):
    def test_elements_from_masses(self):
        # Only test lower masses. The high ones are not very exact
        masses_to_test = _all_masses[_all_masses < 140]
        elements_to_test = _all_elements[_all_masses < 140]
        assert np.array_equal(elements_to_test, elements_from_masses(masses_to_test))
        assert np.array_equal(
            elements_to_test, elements_from_masses(masses_to_test + 0.05)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
