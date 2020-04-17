import unittest

from moleculekit.molecule import Molecule


class Test(unittest.TestCase):

    def test_collision(self):
        mol = Molecule()
        mol1 = Molecule("3ptb")
        mol.append(mol1, collisions=True)

    def test_collision_pass(self):
        mol = Molecule()
        mol1 = Molecule("3ptb")
        mol.append(mol1)
