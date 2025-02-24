import networkx as nx


def plot(G, with_labels: bool = True, seed: int = 1234):
    acts = {i: f"{node.get("act")}_{i}" for i, node in G.nodes(data=True)}
    nx.draw(
        G,
        nx.spring_layout(G, weight="duration", seed=seed),
        labels=acts,
        with_labels=with_labels,
        connectionstyle="arc3,rad=0.1",
    )
