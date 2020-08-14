import argparse

import openmesh as om
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', type=str, default='asset/bunny.obj')
parser.add_argument('--save_path', type=str, default='asset/bunny_local_smooth.obj')
parser.add_argument('--normal_iter', type=int, default=20)
parser.add_argument('--vertex_iter', type=int, default=20)
parser.add_argument('--multiple_sigma_c', type=float, default=1.0)
parser.add_argument('--sigma_s', type=float, default=0.23)


def get_face_normal(mesh: om.TriMesh):
    mesh.request_face_normals()
    mesh.update_face_normals()

    normals = np.zeros(shape=(mesh.n_faces(), 3))
    for fh in mesh.faces():
        n = mesh.normal(fh)
        normals[fh.idx()] = n
    return normals


def get_face_centroid(mesh: om.TriMesh):
    centroid = np.zeros(shape=(mesh.n_faces(), 3))
    for fh in mesh.faces():
        c = mesh.calc_face_centroid(fh)
        centroid[fh.idx()] = c
    return centroid


def get_face_area(mesh: om.TriMesh):
    area = np.zeros(shape=(mesh.n_faces(),))
    for fh in mesh.faces():
        p = []
        for vh in mesh.fv(fh):
            p.append(mesh.point(vh))
        e1 = p[1] - p[0]
        e2 = p[1] - p[2]
        # compute face area by cross product
        s = 0.5 * np.linalg.norm(np.cross(e1, e2))
        area[fh.idx()] = s
    return area


def update_sigma_c(mesh: om.TriMesh, args):
    sigma_c = 0.0
    num = 0
    face_centroid = get_face_centroid(mesh)
    for fh in mesh.faces():
        c_i = face_centroid[fh.idx()]
        for fh_j in mesh.ff(fh):
            c_j = face_centroid[fh_j.idx()]
            # update sigma_c by face 's distance
            sigma_c = sigma_c + np.linalg.norm(c_i - c_j)
            num += 1
    sigma_c *= args.multiple_sigma_c / num
    return sigma_c


def update_normal(mesh: om.TriMesh, sigma_c, args):
    face_normal = get_face_normal(mesh)
    face_area = get_face_area(mesh)
    face_centroid = get_face_centroid(mesh)

    filtered_normals = np.zeros_like(face_normal)

    # iter more then once
    for it in range(args.normal_iter):
        # loop over mesh face
        for fh_i in mesh.faces():
            index_i = fh_i.idx()
            n_i = face_normal[index_i]
            c_i = face_centroid[index_i]
            weight_sum = 0.0
            temp_normal = np.zeros_like(n_i)
            # loop neighbor face
            for fh_j in mesh.ff(fh_i):
                index_j = fh_j.idx()
                n_j = face_normal[index_j]
                c_j = face_centroid[index_j]

                # compute w_c
                spatial_distance = np.linalg.norm(c_i - c_j)
                w_c = np.exp(-0.5 * spatial_distance * spatial_distance / (sigma_c * sigma_c))
                # compute w_s
                range_distance = np.linalg.norm(n_i - n_j)
                w_s = np.exp(-0.5 * range_distance * range_distance / (args.sigma_s * args.sigma_s))

                # get weight and update weight_sum
                weight = face_area[index_j] * w_c * w_s
                weight_sum += weight
                # sum normal
                temp_normal += n_j * weight
            # update normal
            temp_normal = temp_normal / weight_sum
            temp_normal = temp_normal / np.linalg.norm(temp_normal)
            filtered_normals[index_i] = temp_normal
        # update all face normal for next iter
        face_normal = filtered_normals

    return filtered_normals


def update_vertex_position(mesh: om.TriMesh, filtered_normals, args):
    new_points = np.zeros(shape=(mesh.n_vertices(), 3))
    # iter more than once
    for it in range(args.vertex_iter):
        face_centroid = get_face_centroid(mesh)
        # loop over vertices
        for vh in mesh.vertices():
            p = mesh.point(vh)
            face_num = 0
            temp_point = np.zeros(shape=(3,))
            # loop over vertex's neighbor face
            for fh in mesh.vf(vh):
                index_j = fh.idx()
                # get face normal
                n_j = filtered_normals[index_j]
                # get face centroid
                c_j = face_centroid[index_j]
                # sum point
                temp_point += n_j * np.dot(n_j, (c_j - p))
                face_num += 1
            # update points
            p = p + temp_point / face_num
            new_points[vh.idx()] = p

    for vh in mesh.vertices():
        mesh.set_point(vh, new_points[vh.idx()])

    return mesh


def bilateral_normal_filtering_local(mesh: om.TriMesh, args):
    sigma_c = update_sigma_c(mesh, args)
    filtered_normals = update_normal(mesh, sigma_c, args)
    new_mesh = update_vertex_position(mesh, filtered_normals, args)

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
    smooth_mesh = bilateral_normal_filtering_local(mesh, args)
    om.write_mesh(args.save_path, smooth_mesh)
