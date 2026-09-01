import base64
import gzip
import struct

import numpy as np
import pytest

from moleculekit.molecule import Molecule
from moleculekit.util import writeCube
from moleculekit.viewer.molstar import render as render_mod
from moleculekit.volume import Volume

needs_chromium = pytest.mark.skipif(
    render_mod._find_chromium_or_none() is None, reason="needs a chromium binary"
)


def _blob(tmp_path, origin=(1.0, 2.0, 3.0), step=0.5, n=16):
    """A gaussian blob written to a cube file, for tests to read back."""
    z, y, x = np.mgrid[0:n, 0:n, 0:n]
    half = (n - 1) / 2
    data = np.exp(-(((z - half) ** 2 + (y - half) ** 2 + (x - half) ** 2) / 8.0))
    path = str(tmp_path / "blob.cube")
    writeCube(data.astype(np.float32), path, np.array(origin), np.array([step] * 3))
    return path, data.astype(np.float32)


def test_reading_a_cube_keeps_the_grid(tmp_path):
    path, data = _blob(tmp_path)
    vol = Volume(path)
    assert vol.shape == data.shape
    assert np.allclose(vol.data, data, atol=1e-5)
    assert np.allclose(vol.spacing, [0.5, 0.5, 0.5])
    assert vol.viewname == "blob.cube"


def test_a_cube_round_trip_stays_in_the_same_place(tmp_path):
    """writeCube takes the grid's minimal corner and writes the first point.

    They are half a voxel apart, so writing what was read drifted the volume
    away from the molecule it belongs beside, a little further every time.
    """
    path, _ = _blob(tmp_path)
    vol = Volume(path)
    again = tmp_path / "again.cube"
    again.write_text(vol.to_cube())
    reread = Volume(str(again))
    assert np.allclose(reread.origin, vol.origin)
    assert np.allclose(reread.spacing, vol.spacing)
    assert np.allclose(reread.data, vol.data, atol=1e-5)


def test_unreadable_formats_say_what_is_supported(tmp_path):
    path = tmp_path / "map.ccp4"
    path.write_bytes(b"not a cube")
    with pytest.raises(RuntimeError, match="Cannot read '.ccp4'"):
        Volume(str(path))


def test_suggested_isovalue_comes_from_the_data(tmp_path):
    """A density's useful contour depends on its units, so it is not a constant."""
    path, data = _blob(tmp_path)
    vol = Volume(path)
    value = vol.suggest_isovalue()
    assert data.min() < value < data.max()
    # A tighter quantile encloses less of the map.
    assert vol.suggest_isovalue(0.9999) >= value


def test_volume_representations_are_its_own(tmp_path):
    path, _ = _blob(tmp_path)
    vol = Volume(path)
    vol.reps.add(isovalue=0.4, color="#66ccff", opacity=0.45)
    vol.reps.add(isovalue=0.8)
    assert [r.isovalue for r in vol.reps.replist] == [0.4, 0.8]
    assert vol.reps.replist[0].opacity == 0.45
    # The second took the default colour rather than the first one's.
    assert vol.reps.replist[1].color != "#66ccff"
    assert "isovalue=0.4" in str(vol.reps)

    vol.reps.remove(0)
    assert [r.isovalue for r in vol.reps.replist] == [0.8]
    vol.reps.remove()
    assert vol.reps.replist == []


def test_an_isosurface_with_no_value_picks_one(tmp_path):
    path, _ = _blob(tmp_path)
    vol = Volume(path)
    vol.reps.add(color="red")
    assert vol.reps.replist[0].isovalue == pytest.approx(vol.suggest_isovalue())


def test_volumes_travel_with_the_objects(tmp_path, monkeypatch):
    path, _ = _blob(tmp_path)
    vol = Volume(path)
    vol.reps.add(isovalue=0.4, color="#66ccff", opacity=0.45)
    mol = Molecule().empty(3)
    mol.element[:] = ["N", "C", "C"]
    mol.name[:] = ["N", "CA", "C"]
    mol.resname[:] = "ALA"
    mol.resid[:] = 1
    mol.coords = np.zeros((3, 3, 1), dtype=np.float32)

    seen = {}

    def _capture(payload, **kwargs):
        seen["p"] = payload
        return b"png"

    monkeypatch.setattr(render_mod, "render_png", _capture)
    render_mod.render([mol, vol], size=(50, 50))

    volumes = seen["p"]["globals"]["volumes"]
    assert len(volumes) == 1
    assert volumes[0]["reps"] == [
        {"isovalue": 0.4, "color": "#66ccff", "opacity": 0.45, "wireframe": False}
    ]
    raw = gzip.decompress(base64.b64decode(volumes[0]["ccp4_gz"]))
    assert raw[52 * 4 : 52 * 4 + 4] == b"MAP "


def test_a_volume_alone_has_nothing_to_frame(tmp_path):
    """The camera is aimed with atom selections, and a grid has no atoms."""
    path, _ = _blob(tmp_path)
    with pytest.raises(ValueError, match="at least one molecule"):
        render_mod.render(Volume(path), size=(50, 50))


@needs_chromium
def test_a_volume_draws_alongside_the_molecule(tmp_path):
    mol = Molecule("3ptb")
    mol.filter("protein")
    mol.reps.add("all", "NewCartoon", "Secondary Structure")

    centre = mol.coords[:, :, 0].mean(axis=0)
    n, step = 20, 1.0
    origin = centre - (n * step) / 2
    gz, gy, gx = np.mgrid[0:n, 0:n, 0:n]
    positions = np.stack([gx, gy, gz], axis=-1) * step + origin
    grid = np.exp(-(np.linalg.norm(positions - centre, axis=-1) ** 2) / 20.0)
    path = str(tmp_path / "density.cube")
    writeCube(grid.astype(np.float32), path, origin, np.array([step] * 3))

    vol = Volume(path)
    vol.reps.add(isovalue=0.5, color="#66ccff", opacity=0.6)

    with_volume = render_mod.render([mol, vol], size=(220, 220))
    without = render_mod.render(mol, size=(220, 220))
    assert with_volume != without, "the isosurface drew nothing"

    # Same data, origin moved: the map's own origin has to place the surface,
    # otherwise a grid drawn beside a protein lands wherever the viewer's
    # default puts it. The camera is framed on the molecule either way.
    moved = Volume(data=vol.data, origin=vol.origin + 8.0, spacing=vol.spacing)
    moved.reps.add(isovalue=0.5, color="#66ccff", opacity=0.6)
    assert render_mod.render([mol, moved], size=(220, 220)) != with_volume
    render_mod.shutdown_for_tests()


def test_a_ccp4_map_puts_each_value_where_the_header_says():
    """The map is read by the viewer, so the axis order has to be the one the
    header promises: x fastest. A transposed grid still parses and still draws
    a surface, just a mirrored one, so this checks the bytes themselves."""
    nx, ny, nz = 4, 5, 6
    data = np.arange(nx * ny * nz, dtype=np.float32).reshape(nx, ny, nz)
    vol = Volume(data=data, origin=[1.25, -2.0, 3.5], spacing=[0.4, 0.5, 0.6])
    raw = vol.to_ccp4()

    assert len(raw) == 1024 + 4 * nx * ny * nz
    extent = struct.unpack_from("<3i", raw, 0)
    assert extent == (nx, ny, nz)
    assert struct.unpack_from("<i", raw, 3 * 4)[0] == 2  # float32
    assert struct.unpack_from("<3f", raw, 49 * 4) == pytest.approx((1.25, -2.0, 3.5))
    # Voxel size is the cell divided by its grid sampling, not the extent.
    cell = struct.unpack_from("<3f", raw, 10 * 4)
    grid = struct.unpack_from("<3i", raw, 7 * 4)
    assert [c / g for c, g in zip(cell, grid)] == pytest.approx([0.4, 0.5, 0.6])

    ix, iy, iz = 3, 1, 4
    offset = 1024 + 4 * (ix + nx * (iy + ny * iz))
    assert struct.unpack_from("<f", raw, offset)[0] == data[ix, iy, iz]


def test_a_ccp4_map_is_smaller_than_the_same_grid_as_text(tmp_path):
    """Why the map is binary: a 200-cell grid as cube text is a payload of a
    hundred megabytes, past what the render server accepts. Base64 because
    that is the form the payload actually carries."""
    path, _ = _blob(tmp_path, n=16)
    vol = Volume(path)
    encoded = base64.b64encode(vol.to_ccp4())
    assert len(encoded) < len(vol.to_cube()) / 2


def _grd(tmp_path, data, first=(-93, 62, -26), step=0.5, angles=(90.0, 90.0, 90.0)):
    """Write a grid the way rDock does, for tests to read back.

    The cell is sampled one step coarser than the extent of points, which is
    what a real rDock grid does: 29 intervals of 0.5 A spanned by 30 points.
    """
    extent = np.array(data.shape)
    sampling = extent - 1
    cell = list(sampling * step) + list(angles)
    last = np.array(first) + sampling
    lines = [
        "RBT FFT GRID",
        "(1F15.10)",
        " ".join(f"{v:8.3f}" for v in cell),
        " ".join(f"{v:4d}" for v in sampling),
        " ".join(f"{v:4d}" for v in [1, *sum(zip(first, last), ())]),
    ]
    # x fastest, as the axis order in the header says.
    lines += [f"{v:15.10f}" for v in data.transpose(2, 1, 0).ravel()]
    path = tmp_path / "cavity.grd"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_reading_an_rdock_grid_puts_it_where_the_indices_say(tmp_path):
    """The grid indices are absolute, so they are what places a cavity next to
    the receptor it was mapped on. Asymmetric extents, so a swapped axis or a
    transposed reshape cannot pass."""
    data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    vol = Volume(_grd(tmp_path, data, first=(-93, 62, -26), step=0.5))

    assert vol.shape == (3, 4, 5)
    assert np.allclose(vol.spacing, [0.5, 0.5, 0.5])
    assert np.allclose(vol.origin, [-46.5, 31.0, -13.0])
    assert np.allclose(vol.data, data)
    assert vol.viewname == "cavity.grd"


def test_a_grid_this_cannot_place_is_refused(tmp_path):
    """Both would silently draw the data somewhere other than where it is."""
    data = np.zeros((3, 4, 5), dtype=np.float32)
    path = _grd(tmp_path, data, angles=(90.0, 90.0, 120.0))
    with pytest.raises(RuntimeError, match="orthogonal"):
        Volume(path)

    path = _grd(tmp_path, data)
    lines = open(path).read().split("\n")
    lines[4] = lines[4].replace("   1", "   2", 1)
    open(path, "w").write("\n".join(lines))
    with pytest.raises(RuntimeError, match="axis order"):
        Volume(path)


def test_updating_a_surface_keeps_the_rest_of_it(tmp_path):
    path, _ = _blob(tmp_path)
    vol = Volume(path)
    vol.reps.add(isovalue=0.4, color="#66ccff", opacity=0.45)
    vol.reps.add(isovalue=0.8)
    vol.reps.update(0, color="#ff6600")

    first = vol.reps.replist[0]
    assert (first.color, first.isovalue, first.opacity) == ("#ff6600", 0.4, 0.45)
    assert [r.isovalue for r in vol.reps.replist] == [0.4, 0.8]


def test_a_hidden_surface_is_left_out_of_the_scene(tmp_path, monkeypatch):
    path, _ = _blob(tmp_path)
    vol = Volume(path)
    vol.reps.add(isovalue=0.4, color="#66ccff")
    vol.reps.add(isovalue=0.8, color="#0044aa")
    vol.reps.update(0, visibility=False)
    assert "hidden" in str(vol.reps)

    mol = Molecule().empty(3)
    mol.element[:] = ["N", "C", "C"]
    mol.name[:] = ["N", "CA", "C"]
    mol.resname[:] = "ALA"
    mol.resid[:] = 1
    mol.coords = np.zeros((3, 3, 1), dtype=np.float32)

    seen = {}

    def _capture(payload, **kwargs):
        seen["p"] = payload
        return b"png"

    monkeypatch.setattr(render_mod, "render_png", _capture)
    render_mod.render([mol, vol], size=(50, 50))

    reps = seen["p"]["globals"]["volumes"][0]["reps"]
    assert [r["isovalue"] for r in reps] == [0.8]
    # Still there to switch back on, just not drawn.
    assert len(vol.reps.replist) == 2
