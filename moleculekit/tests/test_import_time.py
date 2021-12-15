# (c) 2015-2022 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import unittest
import time


class _TestImportTime(unittest.TestCase):
    def test_importTime(self):
        start_time = time.time()
        from moleculekit.molecule import Molecule

        elapsed_time = time.time() - start_time

        assert elapsed_time < 0.5


if __name__ == "__main__":
    unittest.main(verbosity=2)
