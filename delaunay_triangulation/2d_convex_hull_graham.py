import argparse
from collections import deque
import functools

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--point_num', type=int, default=100)

if __name__ == '__main__':
    args = parser.parse_args()
    # sample points
    points = np.random.random(size=(args.point_num, 2))
    plt.scatter(points[:, 0], points[:, 1])

    # sort by x, if x eq sort by y
    points = list(points)
    points.sort(key=lambda x: (x[0], x[1]))

    # smallest point into stack
    s = deque()
    p1 = points.pop(0)
    s.append(p1)


    # sort by polar angle
    def cmp(lhs, rhs):
        angle_lhs = np.arctan2((lhs[1] - p1[1]), (lhs[0] - p1[0]))
        angle_rhs = np.arctan2((rhs[1] - p1[1]), (rhs[0] - p1[0]))
        if angle_lhs > angle_rhs:
            return 1
        else:
            return -1


    points.sort(key=functools.cmp_to_key(cmp))

    # add polar angle min element to s
    s.append(points.pop(0))

    # Loop
    while len(points) != 0:
        t1 = points[0]
        s1 = s[len(s) - 1]
        s2 = s[len(s) - 2]
        # judge side
        if np.cross(s1 - s2, t1 - s2) >= 0:
            s.append(points.pop(0))
        else:
            s.pop()

    # connect head and tail
    s.append(s[0])
    # plot
    s = np.array(s)
    plt.plot(s[:, 0], s[:, 1], color='red')
    plt.show()
