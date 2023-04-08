import numpy as np
import numpy.typing as npt

import coronaryx.functions as cxf
from coronaryx.data import VesselBranch
from coronaryx.algorithms.traverse import traverse_branch_edges


# TODO do we really need width? maybe we should consider masks too? do we need to pass the extrapolation argument?
# TODO rename to sweep_along_branch?
def sweep(
        branch: VesselBranch,
        matrix: np.ndarray['n, m', float],
        sampling_rate: int = 4,
        width: int = 50,
        return_roi_positives: bool = False
) -> np.ndarray['n, width', float] | tuple[np.ndarray['n, width', float], np.ndarray['n', float]]:

    sweeps = []
    positives = []
    for v, w in traverse_branch_edges(branch):
        v = np.array(v, dtype=np.float32)
        w = np.array(w, dtype=np.float32)
        # TODO vvv
        # we could normalize this and omit `sampling_rate`
        # maybe sampling_rate is good? maybe we should get another argument - sampling_strategy
        # it will be either 'norm' or 'edge', norm means sampling_rate is times per unit vector,
        # and edge is times per edge, which will likely result in inconsistent distances between samples
        vessel_direction = w - v
        orthonormal_vector = cxf.orthonormal_2d_vector(vessel_direction)  # TODO make direction deterministic? is it now?

        multipliers = np.arange(width) - width / 2 + 0.5
        sweep_vector = multipliers[:, np.newaxis] * orthonormal_vector

        for i in range(sampling_rate):
            anchor = v + (vessel_direction * i) / sampling_rate
            sweep_points = anchor + sweep_vector
            sweep_cut = cxf.interp_matrix(matrix, sweep_points[:, 0], sweep_points[:, 1])

            sweeps.append(sweep_cut)
            positives.append(branch.scan.in_any_roi(anchor))

    if return_roi_positives:
        return np.array(sweeps), np.array(positives)
    return np.array(sweeps)


# TODO define alias functions for unraveling, width calculation and so on? should we?
