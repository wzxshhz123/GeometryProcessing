import igl
import numpy as np

from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix

from meshplot import subplot


def tutte(v: np.array, f: np.array):
    # compute boundary_vertex
    boundary_vertex = igl.boundary_loop(f)

    # map boundary_vertex to circle, get boundary_vertex 's 2D coordinate
    boundary_vertex_uv = igl.map_vertices_to_circle(v, boundary_vertex)

    # Construct A
    vv = igl.adjacency_list(f)
    A = np.zeros(shape=(v.shape[0], v.shape[0]))

    for i, v_list in enumerate(f):
        for j, v_index in enumerate(v_list):
            if v_index in boundary_vertex:
                continue
            k = (j + 1) % 3
            l = (j + 2) % 3

            # exist edge, coefficients is 1
            A[v_index, f[i, k]] = 1
            A[v_index, f[i, l]] = 1
            # otherwise a_ii = -valence
            A[v_index, v_index] = -len(vv[f[i, j]])

    for boundary_vertex_index in boundary_vertex:
        A[boundary_vertex_index, boundary_vertex_index] = 1

    # Construct b_u, b_v
    b_u = np.zeros(shape=(v.shape[0],))
    b_v = np.zeros(shape=(v.shape[0],))
    for i, boundary_index in enumerate(boundary_vertex):
        b_u[boundary_index] = boundary_vertex_uv[i, 0]
        b_v[boundary_index] = boundary_vertex_uv[i, 1]

    # solve for result
    A = csc_matrix(A)
    u_ = spsolve(A, b_u)
    v_ = spsolve(A, b_v)
    uv = np.concatenate((u_, v_), axis=0)

    uv *= 5
    return uv


if __name__ == '__main__':
    v, f = igl.read_triangle_mesh("../asset/camelhead.off")
    uv = tutte(v, f)
    p = subplot(v, f, uv=uv, shading={"wireframe": False, "flat": False}, s=[1, 2, 0])
