import argparse
from queue import PriorityQueue

import openmesh as om
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', type=str, default='../asset/bunny.obj')
parser.add_argument('--save_path', type=str, default='../asset/bunny_simplification.obj')
parser.add_argument('--target_vertex_num', type=int, default=1500)
parser.add_argument('--cost_min', type=float, default=0.23)


def simplification(mesh: om.TriMesh, args):
    if mesh.n_vertices() <= args.target_vertex_num:
        print("Error, num of vertex less than target")
        exit()
    vertex = np.zeros(shape=(mesh.n_vertices(), 4, 1))
    # init Q
    initial_q = np.zeros(shape=(mesh.n_vertices(), 4, 4))
    for fh in mesh.faces():
        v_list = []
        v_index_list = []
        # get face's vertex
        for vh in mesh.fv(fh):
            v = mesh.point(vh)
            vertex[vh.idx()][0] = v[0]
            vertex[vh.idx()][1] = v[1]
            vertex[vh.idx()][2] = v[2]
            vertex[vh.idx()][3] = 1
            v_list.append(vertex[vh.idx()])
            v_index_list.append(vh.idx())
        v1 = v_list[0]
        v2 = v_list[1]
        v3 = v_list[2]
        p = np.zeros(shape=(4, 1))
        # get a, b, c, d(ax + by + cz + d = 0) by triangle vertex's pos
        p[0] = v1[1] * (v2[2] - v3[2]) + v2[1] * (v3[2] - v1[2]) + v3[1] * (v1[2] - v2[2])
        p[1] = v1[2] * (v2[0] - v3[0]) + v2[2] * (v3[0] - v1[0]) + v3[2] * (v1[0] - v2[0])
        p[2] = v1[0] * (v2[1] - v3[1]) + v2[0] * (v3[1] - v1[1]) + v3[0] * (v1[1] - v2[1])
        p[3] = -(v1[0] * (v2[1] * v3[2] - v3[1] * v2[2]) +
                 v2[0] * (v3[1] * v1[2] - v1[1] * v3[2]) +
                 v3[0] * (v1[1] * v2[2] - v2[1] * v1[2]))

        # get pp
        pp = np.dot(p, p.T)

        # accumulate pp to face's neighbor vertex
        for v_index in v_index_list:
            initial_q[v_index] += pp

    # collect all valid pair
    vertex_pair_list = []
    for vertex_pair in mesh.ev_indices():
        vertex_pair_list.append(vertex_pair)

    # compute optimal pos and cost for every valid pair
    queue = PriorityQueue()
    for i, pair in enumerate(vertex_pair_list):
        v1 = vertex[pair[0]]
        v2 = vertex[pair[1]]
        v_mean = (v1 + v2) / 2
        # compute q
        q = initial_q[pair[0]] + initial_q[pair[1]]

        # TODO: use optimal
        if np.linalg.det(q) != 0:
            optimal = np.dot(np.linalg.inv(q), np.array((0, 0, 0, 1)))
        # compute different cost
        cost_v1 = np.abs(np.dot(np.dot(v1.T, q), v1))
        cost_v2 = np.abs(np.dot(np.dot(v2.T, q), v2))
        cost_mean = np.abs(np.dot(np.dot(v_mean.T, q), v_mean))

        # find min_cost point and according cost
        if cost_v2 < cost_v1:
            pair[0], pair[1] = pair[1], pair[0]
            queue.put((cost_v2, i))  # (cost, pair_index)
        else:
            queue.put((cost_v1, i))  # (cost, pair_index)

    # main loop for iter
    iter_num = mesh.n_vertices() - args.target_vertex_num
    import tqdm
    for i in tqdm.trange(iter_num):
        # get min cost (cost, pair_index)
        top = queue.get()
        pair = vertex_pair_list[top[1]]

        # TODO: optimal collapse edge
        # collapse edge
        vertex[pair[0]] = vertex[pair[1]]

        # TODO: compute and update new cost
        # update cost
        vertex_pair_list.pop(top[1])
        queue = PriorityQueue()
        for i, pair in enumerate(vertex_pair_list):
            v1 = vertex[pair[0]]
            v2 = vertex[pair[1]]
            v_mean = (v1 + v2) / 2
            # compute q
            q = initial_q[pair[0]] + initial_q[pair[1]]
            # compute different cost
            cost_v1 = np.abs(np.dot(np.dot(v1.T, q), v1))
            cost_v2 = np.abs(np.dot(np.dot(v2.T, q), v2))
            cost_mean = np.abs(np.dot(np.dot(v_mean.T, q), v_mean))

            # find min_cost point and according cost
            if cost_v2 < cost_v1:
                pair[0], pair[1] = pair[1], pair[0]
                queue.put((cost_v2, i))  # (cost, pair_index)
            else:
                queue.put((cost_v1, i))  # (cost, pair_index)

    # update vertex pos
    for vh in mesh.vertices():
        mesh.set_point(vh, vertex[vh.idx()].squeeze()[:3])
    return mesh


if __name__ == '__main__':
    # load model and parameter
    args = parser.parse_args()
    mesh = om.read_trimesh(args.load_path)

    # QME Algorithm
    mesh_res = simplification(mesh, args)

    # Save model
    om.write_mesh(args.save_path, mesh_res)
