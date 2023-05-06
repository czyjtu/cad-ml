import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp2d


def interp_matrix(
        matrix: np.ndarray,
        xs: float | np.ndarray['...', float],
        ys: float | np.ndarray['...', float],
        kind: str = 'linear'
) -> float | np.ndarray['...', float]:
    """
    Interpolates matrix values and allows using float indices.

    :param matrix: 2d numpy ndarray
    :param xs: float index or ndarray of float indices (first dimension)
    :param ys: float index or ndarray of float indices (second dimension)
    :param kind: interpolation method: {'linear', 'cubic', 'quintic'}
    :return: interpolated float value or ndarray of the same shape of the interpolated float values
    """

    if matrix.ndim != 2:
        raise ValueError('unable to process non-2d arrays')

    xs = np.array(xs)
    ys = np.array(ys)

    if xs.shape != ys.shape:
        raise ValueError('xs and ys must have the same shape')

    x_idxs = np.arange(matrix.shape[0])
    y_idxs = np.arange(matrix.shape[1])

    # scipy considers x, y in reverse
    interp = interp2d(y_idxs, x_idxs, matrix, kind=kind)

    if xs.shape == tuple():
        return interp(ys, xs)

    # TODO a place for optimization?
    interpolated = []
    for x, y in zip(xs.flatten(), ys.flatten()):
        interpolated.append(interp(y, x))

    # FIXME vectorized way sorts the input, and I don't know how to map it to the original
    # interpolated = np.diag(interp(ys, xs)).reshape(xs.shape)

    return np.array(interpolated).reshape(xs.shape)


def orthonormal_2d_vector(v: np.ndarray['2', float]) -> np.ndarray['2', float]:
    """
    Computes a normalized orthogonal (orthonormal) vector to the specified in the 2d space
    :param v: input vector
    :return: orthonormal vector to `v`
    """

    ov = np.array([v[1], -v[0]], dtype=v.dtype)
    return ov / np.linalg.norm(ov)
