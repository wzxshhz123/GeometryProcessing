import numpy as np
import trimesh


def mass_matrix(mesh: trimesh.Trimesh, type='BARYCENTRIC'):
    if type == 'BARYCENTRIC':
        face_area = mesh.area_faces
        barycentric_area_per_face = face_area / 3.
        M = np.zeros(shape=(mesh.vertices.shape[0],))
        Minv = np.zeros(shape=(mesh.vertices.shape[0],))
        for i, v_near_face in enumerate(mesh.vertex_faces):
            M[i] = np.mean(barycentric_area_per_face[v_near_face[v_near_face != -1]])
            Minv[i] = 1. / M[i]
    elif type == 'VORONOI':
        pass
    return np.diag(M), np.diag(Minv)
