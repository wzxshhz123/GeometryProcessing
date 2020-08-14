import argparse

import openmesh as om
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', type=str, default='/Users/bear/Dropbox/Code/asset/lucy-noisy.obj')
parser.add_argument('--save_path', type=str, default='lucy_bilateral_smooth.obj')
parser.add_argument('--iter', type=int, default=2)
parser.add_argument('--sigma_s', type=float, default=0.23)
parser.add_argument('--sigma_c', type=float, default=0.23)


def get_vertex_normal(mesh: om.TriMesh):
    mesh.request_vertex_normals()
    mesh.update_vertex_normals()

    normals = np.zeros(shape=(mesh.n_faces(), 3))
    for vh in mesh.vertices():
        n = mesh.normal(vh)
        normals[vh.idx()] = n
    return normals


def update_vertex(mesh: om.TriMesh, args):
    vertex_normal = get_vertex_normal(mesh)

    # init pos
    point_pos = np.zeros(shape=(mesh.n_vertices(), 3))
    for vh in mesh.vertices():
        point_pos[vh.idx()] = mesh.point(vh)

    # iter more then once
    for it in range(args.iter):
        # loop over mesh vertex
        temp_points_pos = np.zeros(shape=(mesh.n_vertices(), 3))
        for vh_i in mesh.vertices():
            index_i = vh_i.idx()
            n_i = vertex_normal[index_i]
            k_sum = 0
            normalizer = 0
            for vh_j in mesh.vv(vh_i):
                t = np.linalg.norm(point_pos[vh_i.idx()] - point_pos[vh_j.idx()])
                h = np.dot(n_i, point_pos[vh_i.idx()] - point_pos[vh_j.idx()])
                # args.sigma_s = t
                w_s = np.exp(-h * h / (2 * args.sigma_s * args.sigma_s))
                w_c = np.exp(-t * t / (2 * args.sigma_c * args.sigma_c))
                k_sum = k_sum + w_c * w_s * h
                normalizer = normalizer + w_c * w_s
            temp_points_pos[vh_i.idx()] = point_pos[vh_i.idx()] + n_i * (k_sum / normalizer)
        # update once point_pos for next iter
        point_pos = temp_points_pos

    for vh in mesh.vertices():
        mesh.set_point(vh, point_pos[vh.idx()])
    return mesh


def bilateral_mesh_denoising(mesh: om.TriMesh, args):
    new_mesh = update_vertex(mesh, args)
    return new_mesh


def half_edge_to_vf(mesh: om.TriMesh):
    v = np.ndarray(shape=(mesh.n_vertices(), 3))
    f = np.ndarray(shape=(mesh.n_faces(), 3), dtype=np.int)
    for vh in mesh.vertices():
        v[vh.idx()] = np.array(mesh.point(vh))

    for fh in mesh.faces():
        temp = []
        for vh in mesh.fv(fh):
            temp.append(vh.idx())
        f[fh.idx()] = np.array(temp)
    return v, f


if __name__ == '__main__':
    args = parser.parse_args()
    mesh = om.read_trimesh(args.load_path)
    # Smooth
    smooth_mesh = bilateral_mesh_denoising(mesh, args)
    om.write_mesh(args.save_path, smooth_mesh)
