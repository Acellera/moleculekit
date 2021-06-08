# GNU LGPL v3
# Copyright (C) 2015-2018 Acellera
# info@acellera.com

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import traceback
import sys


def getOpenBabelProperties(pdb, outfile):
    try:
        from openbabel import pybel
    except ImportError:
        print(
            "Could not import openbabel. The atomtyper requires this dependency so please install it with `conda install openbabel -c conda-forge`"
        )
        sys.exit(1)

    try:
        mpybel = next(pybel.readfile("pdb", pdb))
    except Exception:
        traceback.print_exc()
        sys.exit(2)

    try:
        with open(outfile, "w") as f:
            for r in pybel.ob.OBResidueIter(mpybel.OBMol):
                for at in pybel.ob.OBResidueAtomIter(r):
                    f.write(
                        f"{at.GetIndex()},{r.GetName()},{r.GetNum()},{r.GetAtomID(at)},{at.GetType()},{at.GetPartialCharge():.3f}\n"
                    )
    except Exception:
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    getOpenBabelProperties(sys.argv[1], sys.argv[2])
