import numpy as np
import trimesh
import igl
from PIL import Image


def compute_time_step(mesh: trimesh.Trimesh, smooth=1):
    avg_length = np.mean(mesh.edges_unique_length)
    return avg_length ** 2 * smooth


if __name__ == '__main__':
    mesh = trimesh.load_mesh('/Users/bear/Dropbox/Code/asset/bunny.obj', process=False)
    mesh: trimesh.Trimesh

    source_id = 1  # source vid

    # get boundary vertex
    vertex_boundary_is_border = igl.is_border_vertex(mesh.vertices, mesh.faces)
    # get time step
    time_step = compute_time_step(mesh)

    cot = igl.cotmatrix(mesh.vertices, mesh.faces).toarray()
    M = igl.massmatrix(mesh.vertices, mesh.faces).toarray()

    # solve lap equation to get heat u
    A = M - time_step * cot
    b = np.zeros((mesh.vertices.shape[0], 1))
    A[source_id] = 0
    A[source_id, source_id] = 1
    b[source_id] = 1

    u = np.linalg.solve(A, b)
    u[vertex_boundary_is_border] = 0  # Dirichlet condition

    # get X
    G = igl.grad(mesh.vertices, mesh.faces).toarray()
    grad_u = np.dot(G, u).reshape(-1, 3, order='F')
    X = -grad_u / (np.linalg.norm(grad_u, axis=1, keepdims=True) + 1e-10)

    # get Div x
    div = -0.25 * np.dot(G.T, np.diag(np.tile(2 * mesh.area_faces, 3)))
    div_X = np.dot(div, X.reshape(-1, order='F'))

    cot[source_id] = 0
    cot[source_id, source_id] = 1
    div_X[source_id] = 1
    geodesic_dist = np.linalg.solve(cot, div_X)
    geodesic_dist = (geodesic_dist - np.min(geodesic_dist)).squeeze()

    # show result
    material = Image.open('MyColorBar2.png')
    uv = np.zeros(shape=(mesh.vertices.shape[0], 2))
    uv[:, 0] = geodesic_dist / np.max(geodesic_dist)
    texture = trimesh.visual.texture.TextureVisuals(uv=uv, image=material)
    mesh.visual = texture
    mesh.show()
