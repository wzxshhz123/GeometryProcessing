import numpy as np
from scipy.sparse import csc_matrix
import trimesh


def cot_matrix(mesh: trimesh.Trimesh):
    # get face vertex
    face_vert = mesh.vertices[mesh.faces]
    v0, v1, v2 = face_vert[:, 0], face_vert[:, 1], face_vert[:, 2]

    # get square edge length
    A = np.linalg.norm(v1 - v2, axis=1)
    B = np.linalg.norm(v0 - v2, axis=1)
    C = np.linalg.norm(v0 - v1, axis=1)
    A2, B2, C2 = A * A, B * B, C * C
    l2 = np.stack((A2, B2, C2), axis=1)

    # get area
    area = mesh.area_faces
    # compute cot
    cota = np.true_divide(l2[:, 1] + l2[:, 2] - l2[:, 0], area) / 4.0
    cotb = np.true_divide(l2[:, 2] + l2[:, 0] - l2[:, 1], area) / 4.0
    cotc = np.true_divide(l2[:, 0] + l2[:, 1] - l2[:, 2], area) / 4.0
    cot = np.stack((cota, cotb, cotc), axis=1).reshape(-1)
    cot[cot < 1e-7] = 0

    # get L
    ii = mesh.faces[:, [1, 2, 0]]
    jj = mesh.faces[:, [2, 0, 1]]
    idx = np.stack((ii, jj), axis=0).reshape((2, -1))
    L = csc_matrix((cot, idx), shape=(mesh.vertices.shape[0], mesh.vertices.shape[0])).toarray()
    L += L.T
    for i in range(L.shape[0]):  # get diag
        L[i, i] = -np.sum(L[i, mesh.vertex_neighbors[i]])
    L *= 0.5
    return L
