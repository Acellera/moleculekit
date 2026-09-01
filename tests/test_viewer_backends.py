import numpy as np
import pytest

from moleculekit.molecule import Molecule
from moleculekit.viewer.backends import (
    _backends,
    default_viewer,
    get_viewer,
    register_viewer,
    unregister_viewer,
)


class Recorder:
    """A backend that writes down what it was told, and nothing else."""

    def __init__(self):
        self.events = []

    def view(self, mol, name=None):
        self.events.append(("view", name))
        return "handle"

    def representation_added(self, mol, index, params):
        self.events.append(("added", index, params))

    def representation_updated(self, mol, index, params):
        self.events.append(("updated", index, params))

    def representation_removed(self, mol, index):
        self.events.append(("removed", index))


def _mol():
    mol = Molecule().empty(3)
    mol.element[:] = ["N", "C", "C"]
    mol.name[:] = ["N", "CA", "C"]
    mol.resname[:] = "ALA"
    mol.resid[:] = 1
    mol.coords = np.zeros((3, 3, 1), dtype=np.float32)
    return mol


@pytest.fixture
def backend():
    recorder = Recorder()
    register_viewer("recorder", recorder)
    yield recorder
    unregister_viewer("recorder")
    assert "recorder" not in _backends


def test_a_backend_hears_every_change_as_it_happens(backend):
    """A viewer already on screen cannot wait for a scene to be rebuilt."""
    mol = _mol()
    mol.reps.add("all", "VDW", "Name", opacity=0.5)
    mol.reps.update(0, color="Chain")
    mol.reps.remove(0)

    assert [e[0] for e in backend.events] == ["added", "updated", "removed"]
    added = backend.events[0][2]
    assert added["type"] == "spacefill"
    assert added["opacity"] == 0.5
    # The selection travels with the resolved indices: a viewer following a
    # trajectory re-evaluates it per frame, which indices cannot express.
    assert added["sel"] == "all"
    assert added["atom_indices"] == [0, 1, 2]
    assert added["visibility"] is True
    assert added["update_sel_every_frame"] is None
    assert backend.events[1][2]["color"] == {"theme": "chain-id"}


def test_removing_everything_reports_no_index(backend):
    mol = _mol()
    mol.reps.add("all", "VDW", "Name")
    mol.reps.remove()
    assert backend.events[-1] == ("removed", None)


def test_view_dispatches_to_the_registered_backend(backend):
    mol = _mol()
    assert mol.view(viewer="recorder", name="thing") == "handle"
    assert ("view", "thing") in backend.events


def test_one_registered_backend_becomes_the_default(backend):
    """A live viewer being registered says more about where a molecule should
    go than anything found on PATH."""
    assert default_viewer() == "recorder"
    mol = _mol()
    mol.view()
    assert [e[0] for e in backend.events] == ["view"]


def test_an_unknown_viewer_name_says_how_to_add_one():
    mol = _mol()
    with pytest.raises(ValueError, match="register_viewer"):
        mol.view(viewer="nosuchviewer")


def test_a_backend_must_be_able_to_show_a_molecule():
    class Useless:
        pass

    with pytest.raises(ValueError, match="view"):
        register_viewer("useless", Useless())
    assert get_viewer("useless") is None


def test_the_built_in_viewers_cannot_be_taken_over():
    """Registering over 'molstar' would change what every existing script does."""
    with pytest.raises(ValueError, match="built-in"):
        register_viewer("molstar", Recorder())


def test_a_backend_only_hears_what_it_implements():
    class OnlyAdds:
        def view(self, mol, name=None):
            pass

        def representation_added(self, mol, index, params):
            self.seen = index

    partial = OnlyAdds()
    register_viewer("partial", partial)
    try:
        mol = _mol()
        mol.reps.add("all", "VDW", "Name")
        mol.reps.remove()  # no handler for this, and nothing blows up
        assert partial.seen == 0
    finally:
        unregister_viewer("partial")


def test_nothing_is_translated_when_nobody_is_listening():
    """Building a scene walks the list itself, so the notification must not
    resolve selections behind every add."""
    mol = _mol()
    calls = []
    mol.reps._translateMolstar = lambda rep: calls.append(rep)
    mol.reps.add("all", "VDW", "Name")
    assert calls == []
