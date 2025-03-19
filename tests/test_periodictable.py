def _test_elements_from_masses():
    from moleculekit.periodictable import (
        _all_masses,
        _all_elements,
        elements_from_masses,
    )
    import numpy as np

    # Only test lower masses. The high ones are not very exact
    masses_to_test = _all_masses[_all_masses < 140]
    elements_to_test = _all_elements[_all_masses < 140]
    assert np.array_equal(elements_to_test, elements_from_masses(masses_to_test)[0])
    assert np.array_equal(
        elements_to_test, elements_from_masses(masses_to_test + 0.05)[0]
    )
