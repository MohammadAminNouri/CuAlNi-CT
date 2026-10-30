from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from scipy.spatial.transform import Rotation

from cualni_cryst.arpge_hex_ang import export_hex_parent_ang
from cualni_cryst.ebsd_map import EBSDMap,OrientationConvention
from cualni_cryst.ebsd_io import load_ang
from cualni_cryst.lattice import Lattice


def hex_grid():
    xy=np.asarray(
        [(x,float(r)*.8660254) for r in range(5)
         for x in (np.arange(4) if r%2==0 else np.arange(3)+.5)],
        float,
    )
    g0=Rotation.from_euler("xyz",[12,4,-15],degrees=True).as_matrix()
    g1=Rotation.from_euler("xyz",[-21,11,17],degrees=True).as_matrix()
    source=Rotation.from_matrix(np.asarray([g0 if x<2 else g1 for x,_ in xy])).as_euler("ZXZ")
    ebsd=EBSDMap.from_eulers(
        source,np.ones(len(source),int),xy[:,0],xy[:,1],
        convention=OrientationConvention.bunge_crystal_to_sample_radians(),
    )
    ids=np.asarray([0 if x<2 else 1 for x,_ in xy],int)
    ids[3]=-1
    return ebsd,ids,np.asarray([g0,g1])


def test_real_staggered_hexgrid_ang_preserves_phase_mask_and_parent_rotations(tmp_path:Path):
    ebsd,ids,parents=hex_grid()
    source=export_hex_parent_ang(
        ebsd,ids,parents,parent_lattice=Lattice.cubic(3.31),
    )
    assert "# GRID: HexGrid" in source
    assert "# NCOLS_ODD: 4" in source
    assert "# NCOLS_EVEN: 3" in source
    assert "# NROWS: 5" in source
    path=tmp_path/"hex_parent_research.ang"
    path.write_text(source)
    recovered=load_ang(path)
    assert len(recovered.phase_id)==len(ids)
    assert np.array_equal(recovered.phase_id>0,ids>=0)
    assert np.allclose(recovered.x,ebsd.x)
    assert np.allclose(recovered.y,ebsd.y)
    assert np.allclose(recovered.orientations[ids>=0],parents[ids[ids>=0]],atol=1e-7)


def test_hex_grid_geometry_must_not_be_square_or_irregular():
    from dataclasses import replace
    ebsd,ids,parents=hex_grid()
    with pytest.raises(ValueError,match="lattice"):
        export_hex_parent_ang(ebsd,ids,parents,parent_lattice=None)
    badx=np.asarray(ebsd.x).copy()
    badx[3]+=0.2
    with pytest.raises(ValueError,match="X step"):
        export_hex_parent_ang(
            replace(ebsd,x=badx),ids,parents,parent_lattice=Lattice.cubic(3.31),
        )
    badid=ids.copy();badid[0]=9
    with pytest.raises(ValueError,match="parent-pixel"):
        export_hex_parent_ang(
            ebsd,badid,parents,parent_lattice=Lattice.cubic(3.31),
        )
