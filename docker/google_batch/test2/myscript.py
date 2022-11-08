from moleculekit.molecule import Molecule
import sys

infile = sys.argv[1]

mol = Molecule(infile)
mol.write("results.pdb")
