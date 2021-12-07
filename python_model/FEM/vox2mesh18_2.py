from typing import Tuple
import numpy as np

Offset = Tuple[int, int, int]

# TODO: custom D-type for edge-array w/ cross section size


def get_range(size: int, offset: int):
    if offset > 0:
        a = slice(0, size - offset)
        b = slice(offset, size)
    else:
        a = slice(-offset, size)
        b = slice(0, size + offset)
    return a, b


def get_edges(grid: 'np.ndarray[np.float32]', lut: 'np.ndarray[np.int32]', offset: Offset):
    sx, sy, sz = grid.shape
    ox, oy, oz = offset
    AX, BX = get_range(sx, ox)
    AY, BY = get_range(sy, oy)
    AZ, BZ = get_range(sz, oz)
    connectivity = grid[AX, AY, AZ] & grid[BX, BY, BZ]
    connections = np.nonzero(connectivity)  # type: ignore
    A = lut[AX, AY, AZ][connections].flatten()
    B = lut[BX, BY, BZ][connections].flatten()
    # TODO: allow the grid to influence material strength
    # Transpose Edges to have them pair-wise-stacked
    E = np.vstack([A, B]).T
    return E


def vox2mesh18_2(grid: 'np.ndarray[np.float32]'):
    voxels = np.nonzero(grid)  # type: ignore

    # list[x, y, z] per voxel
    vertices = np.vstack(voxels).T

    # vertex_lut (position => vertex-id)
    lut = np.zeros(grid.shape).astype(np.int32)
    lut[voxels] = range(0, len(vertices))

    # Offsets to check
    offsets: list[Offset] = [
        # faces
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        # edges
        (1, 1, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, -1, 0),
        (1, 0, -1),
        # corners
        (1, 1, 1),
    ]

    # list[a, b, t] per edge
    edges = np.vstack([get_edges(grid, lut, offset) for offset in offsets])

    # TODO: theese edges does only have the indices
    # they are missing the cross-section-size

    return vertices, edges, lut
