import argparse

import tqdm
import igl
import numpy as np
import scipy.sparse.linalg as spl
import scipy.sparse as sp

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', type=str, default='/Users/bear/Dropbox/Code/asset/lucy-noisy.obj')
parser.add_argument('--save_path', type=str, default='lucy_laplacian_smooth.obj')
parser.add_argument('--h', help="time step", type=float, default=0.01)
parser.add_argument('--iter_num', type=int, default=10)
parser.add_argument('--lambda_coffe', type=float, default=0.0001)
parser.add_argument('--type', default='explicit', choices=['implicit', 'explicit'])
if __name__ == '__main__':
    # load param
    args = parser.parse_args()
    # load model
    v, _, n, f, _, _ = igl.read_obj(args.load_path)
    # laplacian
    lap_cot = igl.cotmatrix(v, f)
    # iter num
    t = args.iter_num
    # lambda
    lambda_coffe = args.lambda_coffe
    # time step
    h = args.h
    for i in tqdm.trange(t):
        m = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_BARYCENTRIC)
        if args.type == 'implicit':
            # implicit
            A = sp.csc_matrix(m - h * lambda_coffe * lap_cot)
            b = m * v
            v = spl.spsolve(A, b)
        else:
            # explicit
            a = lambda_coffe * spl.inv(m) * lap_cot * h
            I = np.identity(v.shape[0])
            v = (I + a) * v
    igl.write_obj(args.save_path, v, f)
