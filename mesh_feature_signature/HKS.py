import argparse

import numpy as np
import trimesh
import igl

from basic_tool import jet

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', type=str, default='/Users/bear/Dropbox/Code/asset/bunny.obj')
parser.add_argument('--t', type=float, default=20)


def HKS(mesh: trimesh.Trimesh, t=20, show=False):
    cot = -igl.cotmatrix(mesh.vertices, mesh.faces).toarray()
    eig_value, eig_vector = np.linalg.eigh(cot)
    hks = eig_vector ** 2 * np.exp(-eig_value * t)

    hks_vertex = np.sum(hks, axis=1)
    if show:
        colors = jet(hks_vertex)[:, :3]
        mesh.visual.vertex_colors = colors
        mesh.show()
    return hks


if __name__ == '__main__':
    args = parser.parse_args()
    mesh = trimesh.load_mesh(args.load_path, process=False)
    hks = HKS(mesh, args.t)
