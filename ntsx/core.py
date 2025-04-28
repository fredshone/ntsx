import networkx as nx


def check_time(tst, tet):
    if tet < tst:
        # Assume wrapped around midnight
        tet += 1
        print("Warning: found a wrap, adding one to end time")
    return tst, tet


def vertex_set(G, i, **kwargs):
    for k, v in kwargs.items():
        node = G.nodes[i]
        existing = node.get(k)
        if existing is not None and existing != v:
            print(f"Warning: overwriting {k} from {existing} to {v}")
        node[k] = v


def to_nx(data):
    G = nx.MultiDiGraph()
    i = 0
    for iid, person in data.groupby("iid"):
        i = i + 1
        for _, row in person.iterrows():
            i = i + 1
            tst, tet = check_time(row["tst"], row["tet"])
            duration = tet - tst
            G.add_edge(
                i,
                i + 1,
                duration=duration,
                day=row["day"],
                tst=tst,
                tet=tet,
                travel=row["mode"],
                iid=iid,
            )
            vertex_set(G, i, act=row["oact"], location=row["ozone"])
            vertex_set(G, i + 1, act=row["dact"], location=row["dzone"])

    return G
