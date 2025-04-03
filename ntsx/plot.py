import networkx as nx
import matplotlib.pyplot as plt


def plot(G, ax=None, with_labels: bool = True, seed: int = 1234, **kwargs):
    """Plot a networkx graph with spring layout."""
    acts = {i: node.get("act") for i, node in G.nodes(data=True)}
    node_color = [act for act in acts.values()]
    k = kwargs.pop("k", 0.01)
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    return nx.draw(
        G,
        nx.spring_layout(G, weight="duration", seed=seed, k=k),
        labels=acts,
        with_labels=with_labels,
        connectionstyle="arc3,rad=0.1",
        ax=ax,
        node_color=node_color,
        cmap=plt.cm.tab10,
    )
