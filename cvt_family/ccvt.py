import argparse
import math
from types import SimpleNamespace
import heapq

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--number_sites', type=int, default=128)
    parser.add_argument('--number_points', type=int, default=256)
    parser.add_argument('--torus_size', type=int, default=1000)
    return parser.parse_args()


class Sites:
    def __init__(self, ind, capacity, pos):
        self.ind = ind
        self.capacity = capacity
        self.pos = pos
        self.points = []
        self.energy = 0
        self.stable = False
        self.radius = 0

        # every sites has color
        self.rgb = np.random.rand(3, )

    def update_energy(self):
        if len(self.points) == 0:
            self.energy = 0
            return
        points = np.array(self.points)
        self.energy = np.sum(np.linalg.norm(self.pos - points, axis=1) ** 2)

    def update_centroid(self):
        if len(self.points) == 0:
            return
        temp = np.array(self.points)
        centroid_x = np.sum(temp[:, 0]) / temp.shape[0]
        centroid_y = np.sum(temp[:, 1]) / temp.shape[0]
        self.pos = (centroid_x, centroid_y)

    def update_bound(self):
        if len(self.points) == 0:
            return
        self.radius = np.sqrt(np.max(np.linalg.norm(self.pos - np.array(self.points), axis=1) ** 2))


class Entry:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __lt__(self, other):
        if self.a > other.a:
            return True
        else:
            return False


def constant_regular_density_pts(args: SimpleNamespace, debug=False):
    '''
    generate regular P with constant regular
    :param args:
    :param debug:
    :return:
    '''
    n = int(math.sqrt(args.number_points))
    pts = np.array([(x / n * args.torus_size, y / n * args.torus_size) for x in range(n) for y in range(n)])
    if debug:
        plt.plot(pts[:, 0], pts[:, 1], 'o')
        plt.show()
    return pts


def constant_random_density(args: SimpleNamespace, debug=False):
    '''
    generate random P with constant regular
    :param args:
    :param debug:
    :return:
    '''
    pts = np.random.randint(low=0, high=1000, size=(args.number_points, 2))
    if debug:
        plt.plot(pts[:, 0], pts[:, 1], 'o')
        plt.show()
    return pts


def get_random_sites(args: SimpleNamespace, pts: np.ndarray, debug=False):
    '''
    get random sites set S
    :param args:
    :param pts:
    :param debug:
    :return:
    '''
    sites = []
    max_pos = pts.max()
    x = np.random.rand(args.number_sites) * 1000 % max_pos / max_pos * args.torus_size
    y = np.random.rand(args.number_sites) * 1000 % max_pos / max_pos * args.torus_size
    sites_pos = np.stack((x, y), axis=1)
    # capacity = [args.number_points / args.number_sites] * args.number_sites
    over_capacity = pts.shape[0]
    capacity = []
    for i in range(args.number_sites):
        c = over_capacity / (args.number_sites - i)
        capacity.append(c)
        over_capacity -= c
    ind = np.array([i for i in range(len(capacity))])
    # sites = Sites(ind, capacity, sites_pos)
    # sites = np.array([Sites(i, capacity[i], sites_pos[i]) for i in range(len(capacity))], dtype=object)
    sites = [Sites(i, capacity[i], (sites_pos[i][0], sites_pos[i][1])) for i in range(len(capacity))]
    if debug:
        for i, s in enumerate(sites):
            plt.plot(s.pos[0], s.pos[1], 'p', c=s.rgb)
        plt.show()
    return sites


def init_voroni(pts: np.ndarray, sites: np.ndarray, debug=False):
    '''
    init voroni with coherent initialization
    :param pts:
    :param sites:
    :param debug:
    :return:
    '''
    # get all sites pos
    sites_pos = []
    for s in sites:
        sites_pos.append(s.pos)
    # assign p to sites
    for p in pts:
        kd_tree = cKDTree(sites_pos)
        dis, ind = kd_tree.query(p, k=1, workers=32)
        site_ind = None
        for i, s in enumerate(sites):
            if np.linalg.norm(s.pos - kd_tree.data[ind]) < 1e-4:
                site_ind = i
                break
        sites[site_ind].points.append(p)
        if len(sites[site_ind].points) == sites[site_ind].capacity:
            del sites_pos[ind]

    # init energy for every sites
    for s in sites:
        s.update_energy()
        s.update_centroid()
        s.update_bound()
    if debug:
        for i, s in enumerate(sites):
            plt.plot(s.pos[0], s.pos[1], '*', c=s.rgb, markersize=20)
            if len(s.points) == 0:
                continue
            temp = np.array(s.points)
            plt.plot(temp[:, 0], temp[:, 1], 'o', c=s.rgb)
        plt.show()


def select_site_pais(sites: np.ndarray):
    sites_num = len(sites)
    cluster = []
    for i in range(sites_num):
        for j in range(i + 1, sites_num):
            if sites[i].stable and sites[j].stable:
                continue
            # Bounding circle pruning
            if np.linalg.norm(np.array(sites[i].pos) - np.array(sites[j].pos)) > sites[i].radius + sites[j].radius:
                continue
            cluster.append((i, j))
    return cluster


def site_swap(sites_i: Sites, sites_j: Sites):
    changed = False
    heap_i = []
    heap_j = []

    if len(sites_i.points) > len(sites_j.points):
        sites_j, sites_i = sites_i, sites_j
    for i, p in enumerate(sites_i.points):
        heapq.heappush(heap_i, Entry(np.linalg.norm(p - sites_i.pos) ** 2 - np.linalg.norm(p - sites_j.pos) ** 2, i))
    for j, p in enumerate(sites_j.points):
        heapq.heappush(heap_j, Entry(np.linalg.norm(p - sites_j.pos) ** 2 - np.linalg.norm(p - sites_i.pos) ** 2, j))

    while len(heap_i) > 0 and len(heap_j) > 0:
        max_heap_i = heapq.nsmallest(1, heap_i)[0].a
        max_heap_j = heapq.nsmallest(1, heap_j)[0].a
        if max_heap_i + max_heap_j <= 0:
            break
        heap_i_front = heapq.heappop(heap_i)
        heap_j_front = heapq.heappop(heap_j)

        # get ind
        i_ind = heap_i_front.b
        j_ind = heap_j_front.b
        # swap
        sites_i.points[i_ind], sites_j.points[j_ind] = sites_j.points[j_ind], sites_i.points[i_ind]
        changed = True

    if changed:
        sites_i.stable = False
        sites_j.stable = False
        sites_i.update_centroid()
        sites_j.update_centroid()
        sites_i.update_energy()
        sites_j.update_energy()
        sites_i.update_bound()
        sites_j.update_bound()

    return changed


def optimize(sites: np.ndarray, debug=True):
    stable = False
    while not stable:
        stable = True
        cluster = select_site_pais(sites)

        # changed_list = Parallel(n_jobs=10)(delayed(site_swap)(sites[i], sites[j]) for i, j in cluster)
        # if True in changed_list:
        #     stable = False
        for i, j in cluster:
            changed = site_swap(sites[i], sites[j])
            if changed:
                stable = False
        if debug:
            for i, s in enumerate(sites):
                plt.plot(s.pos[0], s.pos[1], '*', c=s.rgb, markersize=20)
                if len(s.points) == 0:
                    continue
                temp = np.array(s.points)
                plt.plot(temp[:, 0], temp[:, 1], 'o', c=s.rgb)
            plt.show()
    return sites


if __name__ == '__main__':
    args = parse_arguments()
    pts = constant_regular_density_pts(args)
    # pts = constant_random_density(args)
    sites = get_random_sites(args, pts)
    init_voroni(pts, sites, True)
    sites = optimize(sites, True)
    # for i, s in enumerate(sites):
    #     plt.plot(s.pos[0], s.pos[1], '*', c=s.rgb, markersize=15)
    # plt.show()
