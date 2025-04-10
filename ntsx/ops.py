from copy import deepcopy
import networkx as nx
from networkx import edge_bfs

from typing import Optional


def anchor_activities(G, acts=["home"]):
    g = G.copy()
    for act in acts:
        g = squash_on_act(g, act)[0]
    return g


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

    g_new = nx.MultiDiGraph()
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


def merge_similar(
    g: nx.MultiDiGraph, origin=None, duration_tolerance=0.2, verbose=False
) -> nx.MultiDiGraph:
    """Iteratively merge similar activities in the graph based on breadth-first search from origin.

    Args:
        g (nx.MultiDiGraph): Input graph
        origin (optional): Origin node identifier. Defaults to None.
        duration_tolerance (float, optional): Duration similarity tolerance. Defaults to 0.2.
        verbose (bool, optional): Verbosity. Defaults to False.

    Returns:
        nx.MultiDiGraph: Output contracted graph.
    """

    while True:
        result = search_similar(g, origin, duration_tolerance=duration_tolerance)
        if result is not None:
            a, b = result
            if verbose:
                print(f"Contacting; {a} and {b}")
            g = nx.identified_nodes(g, a, b, self_loops=True, copy=True)
            for i, data in g.nodes(data=True):
                data.pop("contraction", None)

        else:
            break
    return g


def search_similar(g: nx.MultiDiGraph, origin, duration_tolerance=0.2):
    """Breadth-first search from origin (ignoring direction) activities that can be combined.
    Combining activities is based on the following criteria:
    1. Similar activities
    2. Similar edges (same mode and similar duration)
    Returns first pair of activities that can be combined.
    If none are found returns None.

    Args:
        g (nx.MultiDiGraph): Input graph
        origin: Origin node identifier
        duration_tolerance (float, optional): Duration similarity tolerance. Defaults to 0.2.

    Returns:
        nx.MultiDiGraph: output graph
    """
    for _, v, _, _ in edge_bfs(g, origin, orientation="ignore"):
        out_edges = set(
            [((u, v, k), False) for u, v, k, in g.out_edges(v, keys=True)]
        )  # outgoing
        in_edges = set(
            [((u, v, k), True) for u, v, k in g.in_edges(v, keys=True)]
        )  # incoming
        edges = out_edges | in_edges
        for a, a_reversed in edges:
            for b, b_reversed in edges:
                ua, va, _ = a
                ub, vb, _ = b

                # get outer nodes
                a_outer = va if not a_reversed else ua
                b_outer = vb if not b_reversed else ub

                if a_outer == b_outer:
                    continue

                if not are_similar_activities(g, a_outer, b_outer):
                    continue

                if are_similar_edges(
                    g,
                    a,
                    b,
                    duration_tolerance=duration_tolerance,
                ):
                    return a_outer, b_outer
    return None


def are_similar_activities(G, a, b):
    node_a = G.nodes[a]
    node_b = G.nodes[b]
    if node_a["act"] != node_b["act"]:
        return False
    if node_a["location"] != node_b["location"]:
        return False
    return True


def are_similar_edges(G, a, b, duration_tolerance=0.2):
    """Return true if edges use same mode and have similar durations."""
    ua, va, ka = a
    ub, vb, kb = b
    edge_a = G[ua][va][ka]
    edge_b = G[ub][vb][kb]

    if edge_a["travel"] != edge_b["travel"]:
        return False
    mean_duration = (edge_a["duration"] + edge_b["duration"]) / 2
    abs_diff = abs(edge_a["duration"] - edge_b["duration"])
    if abs_diff / mean_duration > duration_tolerance:
        return False
    return True


def iter_days(G, stop: Optional[int] = None):
    """Iterate over days in the graph. Each day is a subgraph of the original graph."""
    days = set(data["day"] for _, _, data in G.edges(data=True))
    day_min = min(days)
    days_max = max(days)
    if stop is not None:
        days_max = min(days_max, day_min + stop)
    for day in range(day_min, days_max + 1):
        edges = set(
            [
                (u, v, k)
                for u, v, k, data in G.edges(keys=True, data=True)
                if data["day"] == day
            ]
        ) | set(
            [
                (u, v, k)
                for u, v, k, data in G.edges(keys=True, data=True)
                if data["day"] == day
            ]
        )
        g = G.edge_subgraph(edges)
        if g.number_of_edges() > 0:
            yield day, g


def iter_days_masked(G):
    default_mask = G.copy()
    for u, v, k in default_mask.edges(keys=True):
        default_mask[u][v][k]["masked"] = False

    for day, g in iter_days(G):
        mask_g = default_mask.copy()
        for u, v, k in g.edges(keys=True):
            mask_g[u][v][k]["masked"] = True
        yield day, mask_g
