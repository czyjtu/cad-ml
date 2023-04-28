import coronaryx as cx

import numpy as np
from skimage.segmentation import chan_vese


def chan_vese_mask(
        item: cx.CoronagraphyScan, radius: int = 10,
        mu: float = 0.25, lambda1: int = 1, lambda2: int = 1, tol: float = 1e-4,
        max_num_iter: int = 200, dt: float = 0.1, init_level_set: str = "disk"
    ):
    """
    Segment element-wise using the Chan-vese segmentation

    Parameters
    ----------
    item : cx.CoronagraphyScan
    radius : int, optional
        The radius or size of the segmented element, by default 10
    mu : float, optional
        'edge length' weight parameter. Higher `mu` values will
        produce a 'round' edge, while values closer to zero will
        detect smaller objects.
    lambda1 : float, optional
        'difference from average' weight parameter for the output
        region with value 'True'. If it is lower than `lambda2`, this
        region will have a larger range of values than the other.
    lambda2 : float, optional
        'difference from average' weight parameter for the output
        region with value 'False'. If it is lower than `lambda1`, this
        region will have a larger range of values than the other.
    tol : float, positive, optional
        Level set variation tolerance between iterations. If the
        L2 norm difference between the level sets of successive
        iterations normalized by the area of the image is below this
        value, the algorithm will assume that the solution was
        reached.
    max_num_iter : uint, optional
        Maximum number of iterations allowed before the algorithm
        interrupts itself.
    dt : float, optional
        A multiplication factor applied at calculations for each step,
        serves to accelerate the algorithm. While higher values may
        speed up the algorithm, they may also lead to convergence
        problems.
    init_level_set : str or (M, N) ndarray, optional
        Defines the starting level set used by the algorithm.
        If a string is inputted, a level set that matches the image
        size will automatically be generated. Alternatively, it is
        possible to define a custom level set, which should be an
        array of float values, with the same shape as 'image'.
        Accepted string values are as follows.

        'checkerboard'
            the starting level set is defined as
            sin(x/5*pi)*sin(y/5*pi), where x and y are pixel
            coordinates. This level set has fast convergence, but may
            fail to detect implicit edges.
        'disk'
            the starting level set is defined as the opposite
            of the distance from the center of the image minus half of
            the minimum value between image width and image height.
            This is somewhat slower, but is more likely to properly
            detect implicit edges.
        'small disk'
            the starting level set is defined as the
            opposite of the distance from the center of the image
            minus a quarter of the minimum value between image width
            and image height.

    Returns
    -------
    npt.NDArray
        The raw segmentation mask with votes. TODO extract positive values
    """

    xlim = (0, item.scan.shape[0])
    ylim = (0, item.scan.shape[1])

    mask = np.zeros_like(item.scan)
    nodes = [node for node in item.centerline.nodes()]
    for x, y in zip(*list(zip(*nodes))):
        
        # Define ROI (within the image)
        top = max(xlim[0], x - radius)
        bottom = min(xlim[1], x + radius)
        left = max(ylim[0], y - radius)
        right = min(ylim[1], y + radius)

        # Cut ROI to perform segmentation
        roi = item.scan[top:bottom, left:right]

        # Perform segmentation
        seg = chan_vese(
            roi, mu=mu, lambda1=lambda1, lambda2=lambda2, tol=tol,
            max_num_iter=max_num_iter, dt=dt, init_level_set=init_level_set
        )
        seg = seg

        # Paste segmentation result on to mask
        mask[top:bottom, left:right] += seg

    return mask