import argparse

import numpy as np
from scipy.spatial import Delaunay

import openmesh as om
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--point_num', type=int, default=100)
parser.add_argument('--iter_num', type=int, default=1000)


# compute triangle circumcircle center
def compute_circum_center(v_list):
    x1 = v_list[0][0]
    y1 = v_list[0][1]
    x2 = v_list[1][0]
    y2 = v_list[1][1]
    x3 = v_list[2][0]
    y3 = v_list[2][1]

    k = (y1 - y2) / (x1 - x2)
    k1 = -1 / k
    b1 = (x1 + x2) / 2 / k + (y1 + y2) / 2
    k = (y1 - y3) / (x1 - x3)
    k2 = -1 / k
    b2 = (x1 + x3) / 2 / k + (y1 + y3) / 2

    X = (b2 - b1) / (k1 - k2)
    Y = k1 * X + b1
    return np.array((X, Y))


# compute triangle area
def compute_area(v_list):
    v1 = v_list[0]
    v2 = v_list[1]
    v3 = v_list[2]

    return 0.5 * np.abs(np.cross(v2 - v1, v3 - v1))


# quality = inter_radius / exter_radius
def compute_inter_radius_div_exter_radius(v_list):
    v1 = v_list[0]
    v2 = v_list[1]
    v3 = v_list[2]

    a = np.linalg.norm(v3 - v2)
    b = np.linalg.norm(v3 - v1)
    c = np.linalg.norm(v2 - v1)

    exter_radius = (a * b * c) / (4.0 * compute_area(v_list))
    inter_radius = (a + b - c) / 2.0

    return inter_radius / exter_radius


if __name__ == '__main__':
    # load parameter
    args = parser.parse_args()

    # sample points
    points = np.random.random((args.point_num, 2))

    # original res
    tri = Delaunay(points)
    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.show()

    last_quality = 0
    for it in range(args.iter_num):
        # delaunay
        tri = Delaunay(points)

        # Construct mesh
        mesh = om.TriMesh()
        vh_list = []
        for v in points:
            vh_list.append(mesh.add_vertex(np.array((v[0], v[1], 0))))
        for f in tri.simplices:
            mesh.add_face(vh_list[f[0]], vh_list[f[1]], vh_list[f[2]])

        # compute mesh metric
        quality = 0
        for fh in mesh.faces():
            v_list = []
            for v in mesh.fv(fh):
                v_list.append(mesh.point(v)[:2])
            quality += compute_inter_radius_div_exter_radius(v_list)
        quality /= mesh.n_faces()
        # convergence? end : continue
        if np.abs(quality - last_quality) < 1e-8:
            print("iter_num:", it)
            print("quality:", quality)
            print("Good enough, stop iter")
            break
        last_quality = quality

        # Loop update point pos
        new_pos = np.zeros(shape=(args.point_num, 2), dtype=np.float64)
        for vh in mesh.vertices():
            area_sum = 0.0
            value = np.zeros(shape=(2,))

            # fix boundary point
            if mesh.is_boundary(vh):
                new_pos[vh.idx()] = mesh.point(vh)[:2]
                continue

            # get 1-ring face
            for fh in mesh.vf(vh):
                v_list = []
                for v in mesh.fv(fh):
                    v_list.append(mesh.point(v)[:2])

                # get face area and circum_center
                circum_center = compute_circum_center(v_list)
                area = compute_area(v_list)

                # accumulate
                value += circum_center * area
                area_sum += area

            # get new pos
            new_pos_i = value / area_sum
            new_pos[vh.idx()] = new_pos_i

        # update position
        points = new_pos

    # final res
    tri = Delaunay(points)
    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.show()
