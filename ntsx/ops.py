from copy import deepcopy
import networkx as nx


def squash_on_act(G, act="home"):
    anchor_nodes = [i for i, node in G.nodes(data=True) if node.get("act") == act]
    if not anchor_nodes:
        print(f"Could not find any {act} activities.")
        return deepcopy(G), False
    if len(anchor_nodes) == 1:
        print(f"Only one {act} node.")
        return deepcopy(G), False

    locations = set(G[i].get("zone") for i in anchor_nodes)
    if len(locations) > 1:
        print(f"Found multiple locations for {act}: {locations}. UNKNOWN BEHAVIOUR")

    g_new = nx.MultiDiGraph(iid=G.graph["iid"])
    anchor = anchor_nodes[0]

    # add nodes
    for i, data in G.nodes(data=True):
        if i in anchor_nodes:
            i = anchor
        g_new.add_node(i, **data)

    # add edges
    for u, v, data in G.edges(data=True):
        if u in anchor_nodes:
            u = anchor
        if v in anchor_nodes:
            v = anchor
        g_new.add_edge(u, v, **data)

    return g_new, True


def are_similar_activities(G, i, j):
    node_i = G.nodes[i]
    node_j = G.nodes[j]
    if node_i["act"] != node_j["act"]:
        return False
    if node_i["location"] != node_j["location"]:
        return False
    return True


def are_similar_edges(G, ij, uv, duration_tolerance=0.1):
    """Return true if edges use same mode and have similar durations."""
    i, j = ij
    u, v = uv
    edge_a = G[i][j]  # todo: deal with multiple edges
    edge_b = G[u][v]
    if edge_a["travel"] != edge_b["travel"]:
        return False
    mean_duration = (edge_a["duration"] + edge_b["duration"]) / 2
    abs_diff = abs(edge_a["duration"] - edge_b["duration"])
    if abs_diff / mean_duration > duration_tolerance:
        return False
    return True


def merge_similar(new_G, anchors=None, duration_tolerance=0.2, respect_direction=True):
    new_G = new_G.copy()
    if anchors is None:
        anchors = new_G.nodes()

    if respect_direction:
        found = find_similar_with_direction(new_G, anchors, duration_tolerance)
    else:
        found = find_similar_without_direction(new_G, anchors, duration_tolerance)

    while found:
        i, j = found.pop()
        # check if i and j are still in the graph
        if i not in new_G or j not in new_G:
            continue
        nx.contracted_nodes(new_G, i, j, self_loops=True)
    return new_G


def find_similar_with_direction(G, anchors=None, duration_tolerance=0.2):
    found = []
    # first identify modes with shared anchor
    for anchor in anchors:
        successors = list(G.successors(anchor))
        # iterate through all pairs of successors
        for i in successors:
            for j in successors:
                if i == j:
                    continue

                if are_similar_activities(G, i, j) and are_similar_edges(
                    G,
                    (anchor, i),
                    (anchor, j),
                    duration_tolerance=duration_tolerance,
                ):
                    found.append((i, j))

        predecessors = list(G.predecessors(anchor))
        # iterate through all pairs of predecessors
        for i in predecessors:
            for j in predecessors:
                if i == j:
                    continue

                if are_similar_activities(G, i, j) and are_similar_edges(
                    G,
                    (i, anchor),
                    (j, anchor),
                    duration_tolerance=duration_tolerance,
                ):
                    found.append((i, j))
    return found


def find_similar_without_direction(G, anchors=None, duration_tolerance=0.2):
    found = []
    # first identify modes with shared anchor
    for anchor in anchors:
        out_edges = set([(u, v, False) for u, v in G.out_edges(anchor)])
        in_edges = set([(u, v, True) for u, v in G.in_edges(anchor)])  # reversed edges
        edges = out_edges | in_edges
        print(edges)

        for i, j, ij_reversed in edges:
            for u, v, uv_reversed in edges:
                ij = (i, j)
                uv = (u, v)
                i, j = (i, j) if not ij_reversed else (j, i)
                u, v = (u, v) if not uv_reversed else (v, u)
                if j == v:
                    continue

                if are_similar_activities(G, j, v) and are_similar_edges(
                    G,
                    ij,
                    uv,
                    duration_tolerance=duration_tolerance,
                ):
                    found.append((j, v))
    return found


def iter_days(G):
    days = set(data["tst"].days for (_), data in G.edges(data=True)) | set(
        data["tet"].days for (_), data in G.edges(data=True)
    )
    day_min = min(days)
    days_max = max(days)
    for day in range(day_min, days_max + 1):
        edges = set(
            [e for e, data in G.edges(data=True) if data["tst"].days == day]
        ) | set([e for e, data in G.edges(data=True) if data["tet"].days == day])
        yield day, G.edge_subgraph(edges).copy()
