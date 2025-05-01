from ntsx.nts_to_nx import check_time, set_node_attributes, to_individuals_nx, to_nx
import networkx as nx
import pandas as pd

def test_check_time():
    """
    Test the check_time function.
    """
    # test with valid time values
    tst, tet = check_time(0.1, 0.2)
    assert tst == 0.1, f"Expected 0.1, but got {tst}"
    assert tet == 0.2, f"Expected 0.2, but got {tet}"

    # test that a day is added if end time is less than start time
    tst, tet = check_time(0.5, 0.2)
    assert tst == 0.5, f"Expected 0.5, but got {tst}"
    assert tet == 1.2, f"Expected 1.2, but got {tet}"

def test_set_node_attributes():
    G = nx.MultiDiGraph()
    G.add_node(1, attr1="value1", attr2="value2")
    G.add_node(2, attr1="value3", attr2="value4")

    set_node_attributes(G, 1, attr1="new_value1", attr3="value7")
    set_node_attributes(G, 2, attr1="new_value3", attr2="new_value4")

    assert G.nodes[1]["attr1"] == "new_value1", f"Expected 'new_value1', but got {G.nodes[1]['attr1']}"
    assert G.nodes[1]["attr3"] == "value7", f"Expected 'value7', but got {G.nodes[1]['attr3']}"
    assert G.nodes[2]["attr1"] == "new_value3", f"Expected 'new_value3', but got {G.nodes[2]['attr1']}"
    assert G.nodes[2]["attr2"] == "new_value4", f"Expected 'new_value4', but got {G.nodes[2]['attr2']}"
    assert not hasattr(G.nodes[2], "attr3"), f"Expected expecter no attribute, but got {G.nodes[2]['attr3']}"
    assert G.nodes[1]["attr2"] == "value2", f"Expected 'value2', but got {G.nodes[1]['attr2']}"

def test_to_nx(dummy_trip_data):
    trip_data = dummy_trip_data.copy()

    G = to_nx(trip_data)

    assert isinstance(G, nx.MultiDiGraph), "Expected a MultiDiGraph"
    assert len(G.nodes) == 6, f"Expected 6 nodes, but got {len(G.nodes)}"
    assert len(G.edges) == 4, f"Expected 4 edges, but got {len(G.edges)}"
    assert G.nodes[2]["act"] == "home", f"Expected 'home', but got {G.nodes[2]['act']}"
    assert G.nodes[2]["location"] == "zone1", f"Expected 'zone1', but got {G.nodes[2]['location']}"
    assert G.nodes[3]["act"] == "work", f"Expected 'work', but got {G.nodes[3]['act']}"
    assert G.nodes[3]["location"] == "zone2", f"Expected 'zone2', but got {G.nodes[3]['location']}"
    assert G.nodes[5]["act"] == "home", f"Expected 'home', but got {G.nodes[5]['act']}"
    assert G.nodes[5]["location"] == "zone3", f"Expected 'zone3', but got {G.nodes[5]['location']}"
    assert G.nodes[6]["act"] == "work", f"Expected 'work', but got {G.nodes[6]['act']}"
    assert G.nodes[6]["location"] == "zone4", f"Expected 'zone4', but got {G.nodes[6]['location']}"
    assert G.edges[2, 3, 0]["duration"] == 0.1, f"Expected 0.1, but got {G.edges[2, 3, 0]['duration']}"
    assert G.edges[2, 3, 0]["day"] == 1, f"Expected 1, but got {G.edges[2, 3, 0]['day']}"
    assert G.edges[2, 3, 0]["travel"] == "car", f"Expected 'car', but got {G.edges[2, 3, 0]['travel']}"
    assert G.edges[2, 3, 0]["tst"] == 0.1, f"Expected 0.1, but got {G.edges[2, 3, 0]['tst']}"


def test_to_individuals_nx(dummy_trip_data, dummy_attribute_data):
    trip_data = dummy_trip_data.copy()

    individuals = to_individuals_nx(trip_data)

    assert len(individuals) == 2, f"Expected 2 individuals, but got {len(individuals)}"
    assert isinstance(individuals[0], nx.MultiDiGraph), f"Expected a MultiDiGraph for individual 1"
    assert isinstance(individuals[1], nx.MultiDiGraph), f"Expected a MultiDiGraph for individual 2"
    assert individuals[0].graph["iid"] == 1, f"Expected iid=1 for individual 1"
    assert individuals[1].graph["iid"] == 2, f"Expected iid=2 for individual 2"
    assert len(individuals[0].nodes) == 3, f"Expected 3 nodes for individual 1, but got {len(individuals[0].nodes)}"
    assert len(individuals[1].nodes) == 3, f"Expected 3 nodes for individual 2, but got {len(individuals[1].nodes)}"
    assert len(individuals[0].edges) == 2, f"Expected 2 edges for individual 1, but got {len(individuals[0].edges)}"
    assert len(individuals[1].edges) == 2, f"Expected 2 edges for individual 2, but got {len(individuals[1].edges)}"
    assert individuals[0].nodes[1]["act"] == "home", f"Expected 'home', but got {individuals[0].nodes[1]['act']}"
    assert individuals[0].nodes[1]["location"] == "zone1", f"Expected 'zone1', but got {individuals[0].nodes[1]['location']}"
    assert individuals[0].nodes[2]["act"] == "work", f"Expected 'work', but got {individuals[0].nodes[2]['act']}"
    assert individuals[0].nodes[2]["location"] == "zone2", f"Expected 'zone2', but got {individuals[0].nodes[2]['location']}"
    assert individuals[1].nodes[1]["act"] == "home", f"Expected 'home', but got {individuals[1].nodes[1]['act']}"
    assert individuals[1].nodes[1]["location"] == "zone3", f"Expected 'zone3', but got {individuals[1].nodes[1]['location']}"
    assert individuals[1].nodes[2]["act"] == "work", f"Expected 'work', but got {individuals[1].nodes[2]['act']}"
    assert individuals[1].nodes[2]["location"] == "zone4", f"Expected 'zone4', but got {individuals[1].nodes[2]['location']}"
    assert individuals[0].edges[1, 2, 0]["duration"] == 0.1, f"Expected 0.1, but got {individuals[0].edges[1, 2, 0]['duration']}"
    assert individuals[0].edges[1, 2, 0]["day"] == 1, f"Expected 1, but got {individuals[0].edges[1, 2, 0]['day']}"
    assert individuals[0].edges[1, 2, 0]["travel"] == "car", f"Expected 'car', but got {individuals[0].edges[1, 2, 0]['travel']}"
    assert individuals[0].edges[1, 2, 0]["tst"] == 0.1, f"Expected 0.1, but got {individuals[0].edges[1, 2, 0]['tst']}"
    assert individuals[1].edges[1, 2, 0]["duration"] == 0.2, f"Expected 0.1, but got {individuals[1].edges[1, 2, 0]['duration']}"
    assert individuals[1].edges[1, 2, 0]["day"] == 2, f"Expected 2, but got {individuals[1].edges[1, 2, 0]['day']}"
    assert individuals[1].edges[1, 2, 0]["travel"] == "bike", f"Expected 'bike', but got {individuals[1].edges[1, 2, 0]['travel']}"
    assert individuals[1].edges[1, 2, 0]["tst"] == 0.2, f"Expected 0.2, but got {individuals[1].edges[1, 2, 0]['tst']}"

    attribute_data = dummy_attribute_data.copy()
    attribute_data = attribute_data.set_index("iid")

    individuals = to_individuals_nx(trip_data, attribute_data=attribute_data)
    assert len(individuals) == 2, f"Expected 2 individuals, but got {len(individuals)}"
    assert isinstance(individuals[0], nx.MultiDiGraph), f"Expected a MultiDiGraph for individual 1"
    assert isinstance(individuals[1], nx.MultiDiGraph), f"Expected a MultiDiGraph for individual 2"
    assert individuals[0].graph["iid"] == 1, f"Expected iid=1 for individual 1"
    assert individuals[1].graph["iid"] == 2, f"Expected iid=2 for individual 2"
    assert individuals[0].graph["attribute1"] == "value1", f"Expected 'value1', but got {individuals[0].graph['attribute1']}"
    assert individuals[0].graph["attribute2"] == "value3", f"Expected 'value3', but got {individuals[0].graph['attribute2']}"
    assert individuals[1].graph["attribute1"] == "value2", f"Expected 'value2', but got {individuals[1].graph['attribute1']}"
    assert individuals[1].graph["attribute2"] == "value4", f"Expected 'value4', but got {individuals[1].graph['attribute2']}"

