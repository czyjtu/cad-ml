from typing import Any

import networkx as nx

from coronaryx.data import VesselBranch


def traverse_branch_nodes(branch: VesselBranch) -> list[Any]:
    start_node, end_node = branch.boundary_nodes
    return list(nx.dfs_preorder_nodes(branch.branch, source=start_node))


def traverse_branch_edges(branch: VesselBranch) -> list[tuple[Any, Any]]:
    nodes = traverse_branch_nodes(branch)
    return list(zip(nodes[:-1], nodes[1:]))
