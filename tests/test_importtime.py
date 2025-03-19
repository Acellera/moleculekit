def _test_importTime():
    import time

    start_time = time.time()
    from moleculekit.molecule import Molecule

    elapsed_time = time.time() - start_time

    assert elapsed_time < 0.5
