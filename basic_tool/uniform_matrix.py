import numpy as np
import trimesh


def uniform_matrix(mesh: trimesh.Trimesh):
    D = np.diag(np.array([1. / len(vv) for vv in mesh.vertex_neighbors]))
    return D
