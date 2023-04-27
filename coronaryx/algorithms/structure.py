import networkx as nx

from coronaryx.data import CoronagraphyScan, VesselBranch


def split_into_branches(scan: CoronagraphyScan) -> list[VesselBranch]:
    branches = []
    branching_nodes = []

    G: nx.Graph = scan.centerline.copy()
    for node in G.nodes:
        if len(G[node]) >= 3:
            branching_nodes.append(node)

    G.remove_nodes_from(branching_nodes)

    for component in nx.connected_components(G):
        subgraph = G.subgraph(component).copy()
        branches.append(subgraph)

        # adding the branching nodes to each branch, if they were connected
        for branching_node in branching_nodes:
            for bn_neighbor in scan.centerline[branching_node]:
                if bn_neighbor in component:
                    subgraph.add_edge(branching_node, bn_neighbor)

    return [
        VesselBranch(scan, branch)
        for branch in branches
        if branch.number_of_nodes() >= 2
    ]
