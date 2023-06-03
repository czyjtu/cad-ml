from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import networkx as nx

import coronaryx.functions as cxf


@dataclass
class ROI:
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    form: dict[str, Any] = field(default_factory=dict)

    def __contains__(self, item: tuple[int, int] | np.ndarray["2", float]) -> bool:
        y, x = item
        return self.start_x <= x < self.end_x and self.start_y <= y < self.end_y


@dataclass
class CoronagraphyScan:
    name: str
    scan: np.ndarray["scan_height, scan_width", np.uint8]
    vessel_mask: np.ndarray["scan_height, scan_width", bool]
    centerline: nx.Graph
    rois: list[ROI]

    def in_any_roi(self, item: tuple[int, int] | np.ndarray["2", float]) -> bool:
        return any([item in roi for roi in self.rois])

    def crop_at(
        self, anchor: tuple[int, int], size: int, apply_segmentation_mask: bool = False
    ) -> np.ndarray["size, size", np.uint8]:
        row, col = anchor
        offset = size // 2
        scan = self.scan.copy()
        if apply_segmentation_mask:
            scan[self.vessel_mask == False] = 0.0

        if (
            row - offset < 0
            or col - offset < 0
            or row + offset >= self.scan.shape[0]
            or col + offset >= self.scan.shape[1]
        ):
            # interploation
            # dx = np.arange(size) - size / 2
            # dy = np.arange(size) - size / 2
            # XX, YY = np.meshgrid(dx, dy, indexing="ij")
            # interpolated = cxf.interp_matrix(self.scan, row + XX, col + YY)
            # return interpolated

            # padding
            h, w = scan.shape
            padded = np.zeros(
                (w + 2 * size, h + 2 * size), dtype=self.scan.dtype
            )  # TODO: no need to copy entire image. Make it more efficient
            padded[size:-size, size:-size] = scan
            row += size
            col += size
            return padded[row - offset : row + offset, col - offset : col + offset]

        return scan[row - offset : row + offset, col - offset : col + offset]


@dataclass
class VesselBranch:
    scan: CoronagraphyScan
    branch: nx.Graph

    def __post_init__(self):
        # TODO some kind of deterministic sorting? preferably from the vessel start to the end
        assert (
            self.branch.number_of_nodes() >= 2
        ), f"number of nodes in the branch must be at least 2, current is {self.branch.number_of_nodes()}"
        self._boundary_nodes: tuple[Any, Any] = tuple(
            [node for node in self.branch.nodes if len(self.branch[node]) == 1]
        )
        assert (
            len(self._boundary_nodes) == 2
        ), f"number of boundary nodes is equal to {len(self._boundary_nodes)}, but must be 2"

    @property
    def boundary_nodes(self) -> tuple[Any, Any]:
        return self._boundary_nodes
