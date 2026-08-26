import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest

from moleculekit.molecule import Molecule
from moleculekit.viewer.molstar import render as render_mod

STATIC = Path(__file__).parent.parent / "moleculekit" / "viewer" / "molstar" / "static"


def test_headless_page_is_packaged():
    """The built headless entry ships inside the package."""
    page = STATIC / "headless.html"
    assert page.is_file(), "run `npm run build` in viewer-frontend and commit the output"
    assert "mkHeadless" not in page.read_text(), "the page must load a built asset, not inline the code"


def test_headless_page_references_a_built_asset():
    page = (STATIC / "headless.html").read_text()
    assets = [p.name for p in (STATIC / "assets").iterdir()]
    referenced = [name for name in assets if name in page]
    assert referenced, f"headless.html references none of the built assets: {assets}"


def test_headless_page_pins_the_canvas_container_size():
    """The container must not size itself from the canvas.

    When it does, Mol*'s resize grows the canvas, the canvas grows the
    container, and the drawing buffer creeps up on every render. Since the
    camera's fit is derived from the viewport, that silently rezoomed every
    image: repeated renders of one molecule drifted from 516x473 down to
    432x397 pixels of drawn structure.
    """
    page = (STATIC / "headless.html").read_text()
    assert "position: absolute" in page
    assert "inset: 0" in page


def test_interactive_viewer_page_still_ships():
    """The multi-entry build must not drop the server viewer's page."""
    assert (STATIC / "index.html").is_file()


needs_molviewspec = pytest.mark.skipif(
    importlib.util.find_spec("molviewspec") is None,
    reason="molviewspec not installed (optional 'notebook' dependency)",
)
needs_chromium = pytest.mark.skipif(
    render_mod._find_chromium_or_none() is None,
    reason="no chromium binary found; set MOLECULEKIT_CHROMIUM to enable",
)


def _trypsin():
    """A real protein, read from a local file so the tests need no network."""
    mol = Molecule(str(Path(__file__).parent / "test_molecule" / "3ptb_filtered.pdb"))
    mol.filter("not water")
    return mol


def test_quality_presets_are_the_two_documented_ones():
    assert set(render_mod.QUALITY_PRESETS) == {"fast", "high"}
    assert render_mod.QUALITY_PRESETS["fast"]["occlusion"] is False
    assert render_mod.QUALITY_PRESETS["high"]["occlusion"] is True


def test_unknown_quality_is_rejected():
    mol = Molecule().empty(1)
    mol.coords = np.zeros((1, 3, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown quality"):
        render_mod.render(mol, quality="ultra")


def test_degenerate_size_is_rejected():
    """size=(0, h) must raise, not silently return a stale or blank image."""
    mol = Molecule().empty(1)
    mol.coords = np.zeros((1, 3, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="size"):
        render_mod.render(mol, size=(0, 300))


def test_center_matching_no_atoms_is_rejected():
    """A center selection that misses every atom must raise, not silently
    render the default orientation with no way for the caller to notice."""
    mol = Molecule().empty(1)
    mol.coords = np.zeros((1, 3, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="center"):
        render_mod.render(mol, center="resname NOPE")


@pytest.mark.parametrize("zoom", [0.0, -1.0])
def test_non_positive_zoom_is_rejected(zoom):
    """zoom=0.0 used to raise a bare ZeroDivisionError from build_mvs, and a
    negative zoom was silently accepted; both must be rejected here."""
    mol = Molecule().empty(1)
    mol.coords = np.zeros((1, 3, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="zoom"):
        render_mod.render(mol, zoom=zoom)


def test_gl_backend_order_prefers_hardware_when_a_gpu_is_present(monkeypatch):
    monkeypatch.delenv(render_mod._GL_ENV_VAR, raising=False)
    monkeypatch.setattr(render_mod, "_hardware_gl_present", lambda: True)
    assert render_mod._gl_backend_order() == ["hardware", "software"]


def test_gl_backend_order_skips_hardware_without_a_gpu(monkeypatch):
    """A GPU-less container must not pay for a browser start that cannot work."""
    monkeypatch.delenv(render_mod._GL_ENV_VAR, raising=False)
    monkeypatch.setattr(render_mod, "_hardware_gl_present", lambda: False)
    assert render_mod._gl_backend_order() == ["software"]


@pytest.mark.parametrize("backend", ["hardware", "software"])
def test_gl_backend_order_honours_the_env_var(monkeypatch, backend):
    monkeypatch.setenv(render_mod._GL_ENV_VAR, backend)
    monkeypatch.setattr(render_mod, "_hardware_gl_present", lambda: True)
    assert render_mod._gl_backend_order() == [backend]


def test_gl_backend_order_rejects_an_unknown_backend(monkeypatch):
    monkeypatch.setenv(render_mod._GL_ENV_VAR, "metal")
    with pytest.raises(ValueError, match=render_mod._GL_ENV_VAR):
        render_mod._gl_backend_order()


@pytest.mark.parametrize(
    "renderer,usable",
    [
        ("ANGLE (NVIDIA, NVIDIA GeForce RTX 4050, OpenGL 4.5.0)", True),
        ("ANGLE (Google, SwiftShader Device (Subzero), SwiftShader driver)", True),
        ("NO WEBGL CONTEXT", False),
        ("WEBGL PROBE THREW: Error", False),
        ("", False),
        (None, False),
    ],
)
def test_usable_gl_classifies_renderer_strings(renderer, usable):
    """The fallback hinges on this: a wrong answer either wastes a start or
    returns blank images."""
    assert render_mod._usable_gl(renderer) is usable


def test_hardware_gl_absent_without_dri_render_nodes(monkeypatch, tmp_path):
    """No /dev/dri at all: the shape of a sandboxed container."""
    monkeypatch.setattr(render_mod, "_POSIX", True)
    monkeypatch.setattr(render_mod, "_DRM_DIR", tmp_path / "nonexistent")
    assert render_mod._hardware_gl_present() is False


def test_hardware_gl_absent_when_dri_has_no_render_node(monkeypatch, tmp_path):
    """/dev/dri exists but exposes only a card node, no renderD device."""
    monkeypatch.setattr(render_mod, "_POSIX", True)
    (tmp_path / "card0").touch()
    monkeypatch.setattr(render_mod, "_DRM_DIR", tmp_path)
    assert render_mod._hardware_gl_present() is False


def test_hardware_gl_present_with_a_render_node(monkeypatch, tmp_path):
    monkeypatch.setattr(render_mod, "_POSIX", True)
    (tmp_path / "renderD128").touch()
    monkeypatch.setattr(render_mod, "_DRM_DIR", tmp_path)
    assert render_mod._hardware_gl_present() is True


def test_find_chromium_names_the_env_var_when_missing(monkeypatch):
    monkeypatch.delenv("MOLECULEKIT_CHROMIUM", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="MOLECULEKIT_CHROMIUM"):
        render_mod.find_chromium()


def test_find_chromium_honours_the_env_var(monkeypatch, tmp_path):
    fake = tmp_path / "my-chrome"
    fake.write_text("")
    fake.chmod(0o755)
    monkeypatch.setenv("MOLECULEKIT_CHROMIUM", str(fake))
    assert render_mod.find_chromium() == str(fake)


def test_env_var_pointing_at_nothing_is_rejected(monkeypatch):
    monkeypatch.setenv("MOLECULEKIT_CHROMIUM", "/nonexistent/chrome")
    with pytest.raises(RuntimeError, match="MOLECULEKIT_CHROMIUM"):
        render_mod.find_chromium()


@needs_molviewspec
@needs_chromium
def test_render_returns_png_bytes_of_the_requested_size():
    png = render_mod.render(_trypsin(), size=(400, 300))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image
    import io

    assert Image.open(io.BytesIO(png)).size == (400, 300)
    render_mod.shutdown_for_tests()


@needs_molviewspec
@needs_chromium
def test_render_is_not_a_blank_frame(tmp_path):
    """The failure mode that matters: WebGL dies and every pixel is identical."""
    import io

    from PIL import Image, ImageStat

    out = tmp_path / "trypsin.png"
    assert render_mod.render(_trypsin(), str(out), size=(400, 300)) == str(out)

    image = Image.open(io.BytesIO(out.read_bytes())).convert("RGB")
    assert max(ImageStat.Stat(image).stddev) > 5.0, "render is a uniform image"
    render_mod.shutdown_for_tests()


@needs_molviewspec
@needs_chromium
def test_render_reuses_one_browser_across_calls():
    mol = _trypsin()
    render_mod.render(mol, size=(200, 150))
    first = render_mod._state.process.pid
    render_mod.render(mol, size=(200, 150))
    assert render_mod._state.process.pid == first
    render_mod.shutdown_for_tests()


@needs_molviewspec
@needs_chromium
def test_repeated_renders_are_byte_identical():
    """The same molecule rendered repeatedly must produce the same image.

    Two separate defects broke this: the canvas container grew on every render,
    which rezoomed each image, and the offscreen capture pass took a different
    code path on its first use than on every use after. Both were invisible in
    a single render and only showed up when calling render() more than once in
    a session, so this asserts across several calls rather than two.
    """
    mol = _trypsin()
    images = [render_mod.render(mol, size=(320, 240)) for _ in range(4)]
    assert len(set(images)) == 1, "repeated renders of one molecule differ"
    render_mod.shutdown_for_tests()


@needs_molviewspec
@needs_chromium
def test_render_transparent_has_an_alpha_channel():
    """mode == "RGBA" alone proves nothing: canvas.toDataURL always emits an
    alpha channel. Check that some pixels are actually transparent."""
    import io

    from PIL import Image

    png = render_mod.render(_trypsin(), size=(200, 150), transparent=True)
    image = Image.open(io.BytesIO(png))
    assert image.mode == "RGBA"
    assert image.getchannel("A").getextrema()[0] == 0
    render_mod.shutdown_for_tests()


@needs_molviewspec
@needs_chromium
def test_render_high_quality_survives_and_does_not_wedge_the_render_loop():
    """Exercise the occlusion path (quality="high"), the one path task 5
    found could permanently wedge Mol*'s render loop when SsaoParams were
    incomplete. A load() after it must still resolve."""
    png_hi = render_mod.render(_trypsin(), size=(200, 150), quality="high")
    assert png_hi[:8] == b"\x89PNG\r\n\x1a\n"

    png_fast = render_mod.render(_trypsin(), size=(200, 150), quality="fast")
    assert png_fast[:8] == b"\x89PNG\r\n\x1a\n"
    render_mod.shutdown_for_tests()


def test_molecule_render_delegates_with_its_arguments(monkeypatch):
    """Molecule.render is a thin pass-through, so nothing can drift between them."""
    captured = {}

    def _fake_render(mol, output=None, **kwargs):
        captured["mol"] = mol
        captured["output"] = output
        captured["kwargs"] = kwargs
        return b"png"

    monkeypatch.setattr(render_mod, "render", _fake_render)

    mol = Molecule().empty(1)
    mol.coords = np.zeros((1, 3, 1), dtype=np.float32)
    assert mol.render("out.png", size=(640, 480), rotate="top") == b"png"
    assert captured["mol"] is mol
    assert captured["output"] == "out.png"
    assert captured["kwargs"] == {"size": (640, 480), "rotate": "top"}


@needs_molviewspec
@needs_chromium
def test_molecule_render_writes_a_file(tmp_path):
    out = tmp_path / "mol.png"
    _trypsin().render(str(out), size=(200, 150))
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    render_mod.shutdown_for_tests()
