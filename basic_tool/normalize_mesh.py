import trimesh


def normalize_mesh(file_in, file_out):
    mesh = trimesh.load(file_in)
    bounds = mesh.extents
    if bounds.min() == 0.0:
        return

    # translate to origin
    translation = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    translation = trimesh.transformations.translation_matrix(direction=-translation)
    mesh.apply_transform(translation)

    # scale to unit cube
    scale = 1.0 / bounds.max()
    scale_trafo = trimesh.transformations.scale_matrix(factor=scale)
    mesh.apply_transform(scale_trafo)

    mesh.export(file_out)


if __name__ == '__main__':
    normalize_mesh(
        '/Users/bear/Dropbox/Code/leaf.obj',
        '/Users/bear/Dropbox/Code/leaf_norm.obj'
    )
