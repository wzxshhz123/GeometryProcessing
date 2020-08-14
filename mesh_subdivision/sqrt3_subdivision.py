import argparse

import igl
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', type=str, default='asset/bunny.obj')
parser.add_argument('--save_path', type=str, default='asset/bunny_sqrt3_subdivision.obj')
parser.add_argument('--iter_num', type=int, default=3)


def root3sub(v: np.ndarray, f: np.ndarray):
    face_centroid = igl.barycenter(v, f)
    # init res shape
    v_out = np.zeros(shape=(v.shape[0] + f.shape[0], 3))
    f_out = np.zeros(shape=(f.shape[0] * 3, 3), dtype=np.int)
    v_out[:v.shape[0]] = v
    # f_out[:f.shape[0]] = f

    # add new vertex(centroid)
    for i, face in enumerate(f):
        centroid = face_centroid[i]
        insert_index = v.shape[0] + i

        v_out[insert_index] = centroid
        f_out[i] = np.array([face[0], face[1], insert_index])
        f_out[f.shape[0] + 2 * i] = np.array([face[1], face[2], insert_index])
        f_out[f.shape[0] + 2 * i + 1] = np.array([face[2], face[0], insert_index])

    # update old vertex's pos
    vv = igl.adjacency_list(f)
    for i, v_nb in enumerate(vv):
        n = len(v_nb)
        a_n = (4 - 2 * np.cos(np.pi * 2 / n)) / 9

        v_sum = np.zeros(shape=(3,))
        for v_nb_index in v_nb:
            v_sum = v_sum + v[v_nb_index]

        p = v_out[i]
        v_out[i] = p * (1 - a_n) + a_n / n * v_sum

    # delaunay_triangulation
    edge_length = igl.edge_lengths(v_out, f_out)
    _, f_out = igl.intrinsic_delaunay_triangulation(edge_length, f_out)
    return v_out, f_out


if __name__ == '__main__':
    args = parser.parse_args()
    v_out, f_out = igl.read_triangle_mesh(args.load_path)
    for t in range(args.iter_num):
        v_out, f_out = root3sub(v_out, f_out)
    igl.write_triangle_mesh(args.save_path, v_out, f_out)
