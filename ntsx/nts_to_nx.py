import networkx as nx
from pandas import DataFrame
from typing import Optional, List


def check_time(tst, tet):
    if tet < tst:
        # Assume wrapped around midnight
        tet += 1
        print("Warning: found a wrap, adding one to end time")
    return tst, tet


def set_node_attributes(G, i, **kwargs):
    for k, v in kwargs.items():
        node = G.nodes[i]
        existing = node.get(k)
        if existing is not None and existing != v:
            print(f"Warning: overwriting {k} from {existing} to {v}")
        node[k] = v


def to_nx(trip_data: DataFrame) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    i = 0
    for iid, person in trip_data.groupby("iid"):
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
            set_node_attributes(G, i, act=row["oact"], location=row["ozone"])
            set_node_attributes(G, i + 1, act=row["dact"], location=row["dzone"])

    return G


def to_individuals_nx(
    trip_data: DataFrame, attribute_data: Optional[DataFrame] = None
) -> List[nx.MultiDiGraph]:
    persons = []
    for iid, person in trip_data.groupby("iid"):
        G = nx.MultiDiGraph()
        G.graph["iid"] = iid
        i = 0
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
            )
            set_node_attributes(G, i, act=row["oact"], location=row["ozone"])
            set_node_attributes(G, i + 1, act=row["dact"], location=row["dzone"])

        if attribute_data is not None:
            # Add attributes to the graph
            attributes = attribute_data.loc[iid].to_dict()
            G.graph.update(attributes)

        persons.append(G)

    return persons
