from datetime import timedelta
import networkx as nx


def wrapping_time(day, tst, tet):
    tst = timedelta(days=day, minutes=tst)
    tet = timedelta(days=day, minutes=tet)
    if tet < tst:
        tet += timedelta(days=1)
        print("Warning: found a wrap, adding a day to end time")
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
            tst, tet = wrapping_time(row["day"], row["tst"], row["tet"])
            duration = tet - tst
            G.add_edge(
                i,
                i + 1,
                duration=duration.seconds,
                tst=tst,
                tet=tet,
                travel=row["mode"],
                iid=iid,
            )
            vertex_set(G, i, act=row["oact"], location=row["ozone"])
            vertex_set(G, i + 1, act=row["dact"], location=row["dzone"])

    return G
