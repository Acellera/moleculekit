import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIALS_DIR = REPO_ROOT / "doc" / "source" / "tutorials"

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="Tutorials depend on tools (pdb2pqr, propka, rdkit) reliably available only on Linux CI.",
)


def _tutorial_paths():
    if not TUTORIALS_DIR.is_dir():
        return []
    return sorted(
        p for p in TUTORIALS_DIR.rglob("*.md")
        if p.name != "index.md"
    )


@pytest.mark.parametrize("tutorial_path", _tutorial_paths(), ids=lambda p: str(p.relative_to(TUTORIALS_DIR)))
def _test_tutorial_executes(tutorial_path: Path, tmp_path):
    """Execute a MyST-NB tutorial end-to-end; fail if any non-skipped cell raises."""
    jupytext = pytest.importorskip("jupytext")
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    notebook = jupytext.read(str(tutorial_path))
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
        skip_cells_with_tag="skip-execution",
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        pytest.fail(f"{tutorial_path.relative_to(REPO_ROOT)} failed:\n{exc}")
