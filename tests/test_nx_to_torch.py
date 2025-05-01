import networkx as nx
import pytest
from ntsx.nx_to_torch import build_loader, nx_to_torch_geo
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def test_nx_to_torch_geo(dummy_graphs: list[nx.MultiDiGraph]):
    """
    Test the nx_to_torch_geo function.

    Args:
        dummy_graphs (List[nx.MultiDiGraph]): List of dummy MultiDiGraph objects.
    """
    # load dummy graphs from fixture
    graphs = dummy_graphs

    # convert to torch_geometric data objects
    dataset = nx_to_torch_geo(graphs)

    # check the length of the dataset
    assert len(dataset) == len(graphs), f"Expected {len(graphs)} graphs, but got {len(dataset)}"

    # check the type of the dataset
    assert all(isinstance(data, Data) for data in dataset), "All elements in the dataset should be of type Data"

    # check the first graph in the dataset
    assert dataset[0].num_nodes == len(graphs[0].nodes), f"Expected {len(graphs[0].nodes)} nodes, but got {dataset[0].num_nodes}"
    assert dataset[0].num_edges == len(graphs[0].edges), f"Expected {len(graphs[0].edges)} edges, but got {dataset[0].num_edges}"

def test_nx_to_torch_geo_empty_graph():
    """
    Test the nx_to_torch_geo function with an empty graph.
    """
    # create an empty graph
    empty_graph = nx.MultiDiGraph()

    # convert to torch_geometric data object
    dataset = nx_to_torch_geo([empty_graph])

    # check the length of the dataset
    assert len(dataset) == 1, f"Expected 1 graph, but got {len(dataset)}"

    # check the type of the dataset
    assert isinstance(dataset[0], Data), "The element in the dataset should be of type Data"

    # check the number of nodes and edges in the empty graph
    assert dataset[0].num_nodes == 0, f"Expected 0 nodes, but got {dataset[0].num_nodes}"
    assert dataset[0].num_edges == 0, f"Expected 0 edges, but got {dataset[0].num_edges}"

def test_nx_to_torch_geo_with_features(dummy_graphs_for_gcn, dummy_graphs_for_gat):
    """
    Test the nx_to_torch_geo function with graphs that have features.

    Args:
        dummy_graphs_for_gcn (List[nx.MultiDiGraph]): List of dummy MultiDiGraph objects for GCN.
        dummy_graphs_for_gat (List[nx.MultiDiGraph]): List of dummy MultiDiGraph objects for GAT.
    """
    # load dummy graphs from fixture
    graphs = dummy_graphs_for_gcn + dummy_graphs_for_gat

    # convert to torch_geometric data objects
    dataset = nx_to_torch_geo(graphs)

    # check the length of the dataset
    assert len(dataset) == len(graphs), f"Expected {len(graphs)} graphs, but got {len(dataset)}"

    # check the type of the dataset
    assert all(isinstance(data, Data) for data in dataset), "All elements in the dataset should be of type Data"

    # check the first graph with GCN features
    assert hasattr(dataset[0], "act"), "Expected 'act' feature in the first graph"
    assert hasattr(dataset[0], "location"), "Expected 'location' feature in the first graph"
    assert hasattr(dataset[0], "labels"), "Expected 'labels' feature in the first graph"

    # check the first graph with GAT features
    gat_index = len(dummy_graphs_for_gcn)
    assert hasattr(dataset[gat_index], "act"), "Expected 'act' feature in the first GAT graph"
    assert hasattr(dataset[gat_index], "location"), "Expected 'location' feature in the first GAT graph"
    assert hasattr(dataset[gat_index], "duration"), "Expected 'duration' feature in the first GAT graph"
    assert hasattr(dataset[gat_index], "tst"), "Expected 'tst' feature in the first GAT graph"
    assert hasattr(dataset[gat_index], "tet"), "Expected 'tet' feature in the first GAT graph"
    assert hasattr(dataset[gat_index], "travel"), "Expected 'travel' feature in the first GAT graph"
    assert hasattr(dataset[gat_index], "labels"), "Expected 'labels' feature in the first GAT graph"

def test_build_loader(dummy_graphs: list[nx.MultiDiGraph]):
    """
    Test the build_loader function.

    Args:
        dummy_graphs (List[nx.MultiDiGraph]): List of dummy MultiDiGraph objects.
    """
    # load dummy datasets from fixture
    graphs = dummy_graphs
    dataset = nx_to_torch_geo(graphs)

    # build data loader
    loader = build_loader(dataset, batch_size=2, shuffle=True)

    # check the type of the loader
    assert isinstance(loader, DataLoader), "The loader should be of type DataLoader"

    # check the length of the loader
    assert len(loader) > 0, "The loader should not be empty"

    # check the first batch in the loader
    batch = next(iter(loader))
    assert batch.batch_size == 2, f"Expected batch size of 2, but got {batch.batch_size}"

def test_build_loader_empty_dataset():
    """
    Test the build_loader function with an empty dataset.
    """
    # create an empty dataset
    dataset = []

    # check if ValueError is raised for empty dataset
    with pytest.raises(ValueError):
        build_loader(dataset, batch_size=2, shuffle=True)

def test_build_loader_invalid_batch_size(dummy_graphs: list[nx.MultiDiGraph]):
    """
    Test the build_loader function with an invalid batch size.

    Args:
        dummy_graphs (List[nx.MultiDiGraph]): List of dummy MultiDiGraph objects.
    """
    # create a dummy dataset
    graphs = dummy_graphs
    dataset = nx_to_torch_geo(graphs)

    # check if ValueError is raised for invalid batch size
    with pytest.raises(ValueError):
        build_loader(dataset, batch_size=0, shuffle=True)

    with pytest.raises(ValueError):
        build_loader(dataset, batch_size=-1, shuffle=True)