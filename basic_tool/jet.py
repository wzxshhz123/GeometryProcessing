import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm


def jet(D: np.ndarray, cmap=cm.gist_heat):
    norm = mpl.colors.Normalize(vmin=D.min(), vmax=D.max())
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = m.to_rgba(D)
    return colors
