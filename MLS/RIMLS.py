import numpy as np

import open3d as o3d
from scipy.spatial import KDTree


# spatial weight function
def phi(x, h):
    return (1 - x / h ** 2) ** 4


# grad of phi
def dphi(x, h):
    return -4 * (1 - x / h ** 2) ** 3 / h ** 2


def RIMLS(pts: o3d.geometry.PointCloud, pts_tree: KDTree, x: np.ndarray, h=2, sigma_r=0.5, sigma_n=0.5, max_iter=1e5,
          threshold=1e-4):
    x_origin = x.copy()

    while True:
        i = 0
        while True:
            sumW = 0
            sumGw = 0
            sumF = 0
            sumGF = 0
            sumN = 0
            _, neighbor_pts_ind = pts_tree.query(x, k=30, workers=32)
            neighbor_pts = np.asarray(pts.points)[neighbor_pts_ind]
            neighbor_pts_normal = np.asarray(pts.normals)[neighbor_pts_ind]

            for j, p in enumerate(neighbor_pts):
                # get p normal
                p_normal = neighbor_pts_normal[j]
                # project (x - p) to p normal by dotting
                px = x - p
                fx = np.dot(px, p_normal)
                # compute weight
                alpha = 1
                if i > 0:  # not first iteration
                    alpha = np.exp(-((fx - fx) / sigma_r) ** 2) * np.exp(
                        -(np.linalg.norm(p_normal - grad_f) / sigma_n) ** 2)

                w = alpha * phi(np.linalg.norm(px) ** 2, h)
                grad_w = alpha * 2 * px * dphi(np.linalg.norm(px) ** 2, h)

                sumW += w
                sumGw += grad_w
                sumF += w * fx
                sumGF += grad_w * fx
                sumN += w * p_normal

            f = sumF / sumW
            grad_f = (sumGF - f * sumGw + sumN) / sumW

            # count and stop grad_f update
            i += 1
            if i >= max_iter:
                break
        # project x to surface
        x = x - f * grad_f

        # stop projecting?
        if np.linalg.norm(f * grad_f) < threshold:
            break

    # get SDF
    sdf = np.linalg.norm(x_origin - x)


if __name__ == '__main__':
    mesh = o3d.io.read_triangle_mesh('/Users/bear/Dropbox/Code/asset/bunny.obj')
    mesh: o3d.geometry.TriangleMesh
    pts = mesh.sample_points_poisson_disk(1000, use_triangle_normal=True)
    bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pts.points)
    h = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    # o3d.visualization.draw_geometries([pts, bbox], point_show_normal=True)
    pts_tree = KDTree(np.array(pts.points))
    RIMLS(pts, pts_tree, np.array([1, 0, 0]), h)
