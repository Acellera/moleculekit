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


@needs_chromium
def test_render_returns_png_bytes_of_the_requested_size():
    png = render_mod.render(_trypsin(), size=(400, 300))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image
    import io

    assert Image.open(io.BytesIO(png)).size == (400, 300)
    render_mod.shutdown_for_tests()


@pytest.mark.parametrize("fog", [-1, 101])
def test_out_of_range_fog_is_rejected(fog):
    with pytest.raises(ValueError, match="fog must be between 0 and 100"):
        render_mod._scene_description(_trypsin(), fog=fog)


@pytest.mark.parametrize("clip", [0, -5])
def test_non_positive_clip_is_rejected(clip):
    with pytest.raises(ValueError, match="clip must be a positive distance"):
        render_mod._scene_description(_trypsin(), clip=clip)


@needs_chromium
@pytest.mark.parametrize(
    "style", ["Surf", "QuickSurf", "Points", "Putty", "Labels", "FormalCharges"]
)
def test_added_representation_types_draw(style):
    """Each style Mol* gained must actually draw something.

    An unmapped style used to fall through to ball-and-stick, which drew a
    plausible picture of the wrong representation.
    """
    import io

    from PIL import Image

    mol = _trypsin()
    mol.filter("resid 100 to 103")
    mol.formalcharge[0] = 1
    if style == "FormalCharges":
        # Labels alone draw nothing, so they go on top of a representation.
        mol.reps.add("all", "Licorice", "Name")
    mol.reps.add("all", style, "Name")
    image = Image.open(io.BytesIO(render_mod.render(mol, size=(250, 250))))
    assert (np.asarray(image.convert("L")) < 245).sum() > 100, f"{style} drew nothing"
    render_mod.shutdown_for_tests()


@needs_chromium
def test_an_unusable_colour_is_an_error_not_a_wrong_colour():
    """Every unrecognised colour used to reach Color(NaN) and draw alike.

    A mistyped name and a Mol* theme name that is not one of the mapped ones
    both rendered the same wrong colour, which looks deliberate.
    """
    mol = _trypsin()
    mol.reps.add("all", "VDW", "notacolour")
    with pytest.raises(RuntimeError, match="Not a colour"):
        render_mod.render(mol, size=(60, 60))
    render_mod.shutdown_for_tests()


@needs_chromium
def test_clip_sets_the_slab_thickness_and_is_off_by_default():
    """Framing a selection must not silently cut geometry off.

    Mol* takes both clipping planes from the camera's radius, near at
    distance - radius and far at distance + radius, so leaving that at the
    focus radius left only a slab around whatever ``center`` picked: focusing
    a ligand cut away the protein around it, flat sliced faces and all, and no
    amount of fog brings back geometry that was never drawn. The framing
    distance and the clipping radius are set apart now, so the default draws
    everything and ``clip`` asks for a slab of a chosen thickness, which is
    how you see into a buried pocket.
    """
    import io

    from PIL import Image

    def drawn(png):
        return int((np.asarray(Image.open(io.BytesIO(png)).convert("L")) < 245).sum())

    mol = _trypsin()
    opts = dict(size=(300, 300), center="resid 100", zoom=0.3)
    whole = drawn(render_mod.render(mol, **opts))
    wide = drawn(render_mod.render(mol, clip=15, **opts))
    thin = drawn(render_mod.render(mol, clip=5, **opts))
    assert whole > wide > thin, f"clip did not thin the slab: {whole}, {wide}, {thin}"
    render_mod.shutdown_for_tests()


@needs_chromium
def test_fog_changes_the_image_and_does_not_carry_over():
    """Fog must be settable, and must not leak from one render to the next.

    One browser is reused across render() calls, so a scene saying nothing
    about fog would inherit whatever the previous render set. Mol*'s own
    default strength is applied explicitly instead, which is why the default
    render equals fog=15 and still equals it after other strengths have been
    rendered in between.
    """
    mol = _trypsin()
    default = render_mod.render(mol, size=(300, 300))
    assert render_mod.render(mol, size=(300, 300), fog=0) != default
    assert render_mod.render(mol, size=(300, 300), fog=100) != default
    assert render_mod.render(mol, size=(300, 300)) == default
    render_mod.shutdown_for_tests()


@needs_chromium
def test_opposite_orientations_are_not_the_same_image():
    """`rotate` must honour the sign of its direction.

    Mol*'s getFocus puts the direction and up vectors through
    Vec3.matchDirection, which flips them into the hemisphere the camera
    already looks along, so front and back, left and right, and top and
    bottom each rendered one identical image until the camera position was
    computed from the direction here instead.
    """
    mol = _trypsin()
    names = ("front", "back", "left", "right", "top", "bottom")
    images = {n: render_mod.render(mol, size=(300, 300), rotate=n) for n in names}
    collapsed = [n for n in names if list(images.values()).count(images[n]) > 1]
    assert not collapsed, f"orientations render identically: {collapsed}"
    render_mod.shutdown_for_tests()


@needs_chromium
@pytest.mark.parametrize("segid", ["X", ""])
def test_segids_do_not_change_the_render(segid):
    """A segid must not redraw the structure.

    It is written as label_entity_id and Mol* starts a new chain at every
    entity boundary, so a segid subdividing a chain used to break the cartoon
    there: three interior residues given their own segid collapsed a whole
    beta strand into a coil. Mixing blank and non-blank segids was worse, the
    atoms landed in an entity no _entity row declared and createModel threw.
    """
    mol = _trypsin()
    plain = render_mod.render(mol, size=(400, 400))

    split = mol.copy()
    split.segid[:] = "S"
    split.segid[np.isin(split.resid, [100, 101, 102])] = segid
    assert render_mod.render(split, size=(400, 400)) == plain
    render_mod.shutdown_for_tests()


@needs_chromium
def test_framing_follows_the_requested_size():
    """The camera must fit the image being rendered, not the first one.

    Mol* sizes the canvas from its container and the camera's fit distance
    comes from that canvas, so without a resize per render the framing was
    decided once, by whichever size opened the browser, and every later size
    only cropped or padded it. A 600x1200 render taken after a 1200x900 one
    drew the structure 574 pixels wide in a 600 pixel image, all but touching
    both edges; a 900x900 and a 1200x900 render drew it identically at 516x473.
    """
    from PIL import Image
    import io

    mol = _trypsin()
    for size in [(1200, 900), (600, 1200)]:
        image = Image.open(io.BytesIO(render_mod.render(mol, size=size)))
        box = image.convert("RGB").point(lambda v: 255 if v < 245 else 0)
        box = box.convert("L").getbbox()
        # Fraction of whichever dimension the fit binds on.
        fill = max((box[2] - box[0]) / size[0], (box[3] - box[1]) / size[1])
        assert 0.4 < fill < 0.7, f"{size} drew {box}, filling {fill:.2f}"
    render_mod.shutdown_for_tests()


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


@needs_chromium
def test_render_reuses_one_browser_across_calls():
    mol = _trypsin()
    render_mod.render(mol, size=(200, 150))
    first = render_mod._state.process.pid
    render_mod.render(mol, size=(200, 150))
    assert render_mod._state.process.pid == first
    render_mod.shutdown_for_tests()


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


def test_molecule_render_options_match_the_render_function():
    """Molecule.render spells its options out rather than pointing at render().

    That makes its docstring self-contained, at the cost of duplicating the
    signature, so the defaults are compared here to keep the two from drifting.
    """
    import inspect

    skip = {"self", "mol", "output"}
    method = inspect.signature(Molecule.render).parameters
    function = inspect.signature(render_mod.render).parameters
    assert {k: v.default for k, v in method.items() if k not in skip} == {
        k: v.default for k, v in function.items() if k not in skip
    }


def test_molecule_render_delegates_with_its_arguments(monkeypatch):
    """Molecule.render forwards every option by name, so none can be dropped."""
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
    assert captured["kwargs"]["size"] == (640, 480)
    assert captured["kwargs"]["rotate"] == "top"

    import inspect

    expected = set(inspect.signature(Molecule.render).parameters) - {"self", "output"}
    assert set(captured["kwargs"]) == expected


@needs_chromium
def test_molecule_render_writes_a_file(tmp_path):
    out = tmp_path / "mol.png"
    _trypsin().render(str(out), size=(200, 150))
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    render_mod.shutdown_for_tests()


@needs_chromium
def test_render_needs_no_molviewspec(monkeypatch):
    """The render path must not import molviewspec: the sandbox image does not
    ship it, and dropping that requirement is a goal of this design."""
    import builtins

    real_import = builtins.__import__

    def _forbid(name, *args, **kwargs):
        if name.startswith("molviewspec"):
            raise AssertionError("render() must not import molviewspec")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _forbid)
    png = render_mod.render(_trypsin(), size=(200, 150))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    render_mod.shutdown_for_tests()


@needs_chromium
def test_render_rotate_changes_the_image():
    """The camera step of applyScene must actually move the camera: a render
    with rotate="top" must differ from the default orientation."""
    mol = _trypsin()
    default = render_mod.render(mol, size=(200, 150))
    rotated = render_mod.render(mol, size=(200, 150), rotate="top")
    assert default != rotated
    render_mod.shutdown_for_tests()


@needs_chromium
def test_render_formal_charge_changes_the_image():
    """The labels step of applyScene must actually draw something: a molecule
    with a formal charge must render differently from the same molecule with
    every charge zeroed."""
    mol = _trypsin()
    mol.formalcharge[:] = 0
    neutral = render_mod.render(mol, size=(200, 150))
    mol.formalcharge[0] = 1
    charged = render_mod.render(mol, size=(200, 150))
    assert neutral != charged
    render_mod.shutdown_for_tests()


@needs_chromium
def test_render_zoom_changes_the_image():
    """applyCamera's snapshot must survive Mol*'s automatic camera-fit reset
    triggered by newly committed representations, not be overwritten by it."""
    mol = _trypsin()
    default = render_mod.render(mol, size=(200, 150))
    zoomed = render_mod.render(mol, size=(200, 150), zoom=3.0)
    assert default != zoomed
    render_mod.shutdown_for_tests()


@needs_chromium
def test_render_center_changes_the_image():
    """Same as zoom: the camera's focus target must survive the automatic
    camera-fit reset rather than being silently replaced by it."""
    mol = _trypsin()
    default = render_mod.render(mol, size=(200, 150))
    centered = render_mod.render(mol, size=(200, 150), center="resid 50 to 60")
    assert default != centered
    render_mod.shutdown_for_tests()


@needs_chromium
def test_render_shows_the_ligand_and_ion():
    """Explicit hetero representations (Licorice on ``not protein``) must
    actually draw, not silently resolve to Mol*'s registry default
    representation. A representation-type mismatch once made every
    non-cartoon component resolve to that default, which drew barely
    anything for a ligand/ion selection: measured through the built bundle,
    a fixed camera that keeps the ligand and ion in frame showed only a
    ~270-unit total pixel difference against a cartoon-only render (noise: a
    formal-charge label), against ~2200 once the representation type is
    translated correctly.

    This exercises explicit ``mol.reps`` entries, not the automatic scene's
    builtin ligand/ion/water/branched components; see
    ``test_automatic_scene_shows_the_ligand_and_ion`` for those.

    ``center``/``zoom`` are pinned identically in both renders (same
    underlying atom coordinates too) so an auto-fit framing change cannot be
    mistaken for the ligand/ion actually being visible: without pinning,
    simply dropping the ligand/ion components already changes the auto-fit
    crop regardless of whether they ever drew a pixel, which would make a
    pixel-difference assertion pass for the wrong reason.
    """
    import io

    from PIL import Image, ImageChops

    mol = _trypsin()
    mol.reps.add(sel="protein", style="NewCartoon", color="secondary structure")
    mol.reps.add(sel="not protein", style="Licorice")
    with_hetero = render_mod.render(mol, size=(400, 300), center="protein", zoom=0.7)

    protein_only = _trypsin()
    protein_only.reps.add(sel="protein", style="NewCartoon", color="secondary structure")
    without_hetero = render_mod.render(
        protein_only, size=(400, 300), center="protein", zoom=0.7
    )

    img_with = Image.open(io.BytesIO(with_hetero)).convert("RGB")
    img_without = Image.open(io.BytesIO(without_hetero)).convert("RGB")
    diff = np.asarray(ImageChops.difference(img_with, img_without))
    assert diff.sum() > 2000, "the ligand/ion must draw visibly distinct pixels"
    render_mod.shutdown_for_tests()


@needs_chromium
def test_automatic_scene_shows_the_ligand_and_ion():
    """The automatic scene, the one an empty ``mol.reps`` gives every plain
    ``mol.render()``/``mol.view()`` call, must draw hetero atoms too, not
    just the cartoon.

    Unlike ``test_render_shows_the_ligand_and_ion`` above, ``mol.reps`` is
    left empty here so ``build_scene`` takes its ``_automatic_components``
    path and emits the builtin ``ligand``/``ion``/``water``/``branched``
    components. Those four currently draw nothing (``_bcif_bytes`` writes no
    ``entity``/``chem_comp``/``struct_conn`` categories, so Mol* classifies
    every atom as polymer); hetero atoms survive solely through the
    non-standard-resname component built alongside them (see the comment at
    ``scene.py``'s ``_automatic_components``). This pins that survival so a
    future change to those four builtins does not silently empty every
    default render of its ligand and ion.
    """
    import io

    from PIL import Image, ImageChops

    # Larger than the 200x150 used elsewhere in this file: the ligand/ion are
    # a small fraction of the frame at "center=protein", so a small render
    # leaves too little margin above noise (measured diff sum 915 at 200x150
    # against 2031 at 300x220 for the same two molecules/camera).
    mol = _trypsin()  # protein + MOL (ligand) + Cl- (ion), mol.reps empty
    with_hetero = render_mod.render(mol, size=(300, 220), center="protein", zoom=0.7)

    protein_only = _trypsin()
    protein_only.filter("protein")
    without_hetero = render_mod.render(
        protein_only, size=(300, 220), center="protein", zoom=0.7
    )

    img_with = Image.open(io.BytesIO(with_hetero)).convert("RGB")
    img_without = Image.open(io.BytesIO(without_hetero)).convert("RGB")
    diff = np.asarray(ImageChops.difference(img_with, img_without))
    assert diff.sum() > 1000, (
        "the automatic scene must draw the ligand/ion, not just the cartoon"
    )
    render_mod.shutdown_for_tests()


def test_the_devtools_port_is_chromiums_own_choice():
    """The renderer must not pick the devtools port itself.

    A port probed here has to be released before chromium can bind it, so two
    processes starting a renderer at once both saw the same port free and the
    second attached to the first one's browser. They then cleared each other's
    Mol* state ("Could not find node") and closed each other's browser
    ("websocket closed by peer"), which is how the suite failed under
    pytest-xdist while passing when run alone.
    """
    source = Path(render_mod.__file__).read_text()
    assert "--remote-debugging-port=0" in source
    assert "DevToolsActivePort" in source
