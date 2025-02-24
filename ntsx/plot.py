import networkx as nx
import matplotlib.pyplot as plt


def plot(G, ax=None, with_labels: bool = True, seed: int = 1234, **kwargs):
    """Plot a networkx graph with spring layout."""
    acts = {i: node.get("act") for i, node in G.nodes(data=True)}
    node_color = ["cyan" if act == "home" else "lightgrey" for act in acts.values()]
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    return nx.draw(
        G,
        nx.spring_layout(G, weight="duration", seed=seed, k=1),
        labels=acts,
        with_labels=with_labels,
        connectionstyle="arc3,rad=0.1",
        ax=ax,
        node_color=node_color,
    )
