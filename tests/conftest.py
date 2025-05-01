import networkx as nx
import torch
import pytest
import pandas as pd
from typing import List

@pytest.fixture(scope="session")
def dummy_trip_data() -> pd.DataFrame:
    """
    Generate dummy trip data for testing purposes.

    Args:
        num_trips (int, optional, default 2): The number of trips to generate.
        num_nodes (int, optional, default 2): The number of nodes in each trip.
        num_categories (int, optional, default 2): The number of categories for the features.

    Returns:
        pd.DataFrame: The generated trip data as a DataFrame.
    """
    df = pd.DataFrame({
        "iid": [1, 1, 2, 2],
        "tst": [0.1, 0.5, 0.2, 0.3],
        "tet": [0.2, 0.6, 0.4, 0.5],
        "day": [1, 1, 2, 2],
        "mode": ["car", "bus", "bike", "walk"],
        "oact": ["home", "work", "home", "work"],
        "ozone": ["zone1", "zone2", "zone3", "zone4"],
        "dact": ["work", "home", "work", "home"],
        "dzone": ["zone2", "zone1", "zone4", "zone3"]
    })
    return df

@pytest.fixture
def dummy_attribute_data() -> pd.DataFrame:
    """
    Generate dummy attribute data for testing purposes.

    Args:
        num_nodes (int, optional, default 2): The number of nodes to generate.
        num_categories (int, optional, default 2): The number of categories for the features.

    Returns:
        pd.DataFrame: The generated attribute data as a DataFrame.
    """
    df = pd.DataFrame({
        "iid": [1, 2],
        "attribute1": ["value1", "value2"],
        "attribute2": ["value3", "value4"]
    })
    return df

@pytest.fixture(scope="session")
def dummy_graphs_for_gcn(
    num_graphs: int = 10,
    num_nodes: int | List[int] = 10,
    degree: int | List[int] = 3,
    edge_weight: int | List[int] = 1,
    num_cat: int = 10,
    num_labels: int = 3,
) -> List[nx.MultiDiGraph]:
    """
    Generate dummy graphs for testing purposes.

    Args:
        num_graphs (int, optional, default 10): The number of graphs to generate.
        num_nodes (int|List[int], optional, default 10): The number of nodes in each graph.
            If list, the length should be equal to num_graphs, one for each graph.
        degree (int|List[int], optional, default 3): The degree of each node in the graph.
            If list, the length should be equal to num_graphs, one for each graph.
        edge_weight (int|List[int], optional, default 1): The weight of each edge in the graph.
            If list, the length should be equal to num_graphs, one for each graph
        num_cat (int, optional, default 10): The number of categories for the features.
        num_labels (int, optional, default 3): The number of labels for the graphs.
        
    Returns:
        List[nx.MultiDiGraph]: The list of dummy MultiDiGraph object.
    """
    graphs = generate_dummy_graphs(num_graphs, num_nodes, degree, edge_weight)
    graphs = add_dummy_cat_node_features(graphs, num_cat, feature_name="act")
    graphs = add_dummy_cat_node_features(graphs, num_cat, feature_name="location")
    graphs = add_dummy_labels(graphs, num_labels)

    return graphs

@pytest.fixture(scope="session")
def dummy_graphs_for_gat(
    num_graphs: int = 10,
    num_nodes: int | List[int] = 10,
    degree: int | List[int] = 3,
    edge_weight: int | List[int] = 1,
    num_cat: int = 10,
    num_labels: int = 3,
) -> List[nx.MultiDiGraph]:
    """
    Generate dummy graphs for testing purposes.

    Args:
        num_graphs (int, optional, default 10): The number of graphs to generate.
        num_nodes (int|List[int], optional, default 10): The number of nodes in each graph.
            If list, the length should be equal to num_graphs, one for each graph.
        degree (int|List[int], optional, default 3): The degree of each node in the graph.
            If list, the length should be equal to num_graphs, one for each graph.
        edge_weight (int|List[int], optional, default 1): The weight of each edge in the graph.
            If list, the length should be equal to num_graphs, one for each graph
        num_cat (int, optional, default 10): The number of categories for the features.
        num_labels (int, optional, default 3): The number of labels for the graphs.
        
    Returns:
        List[nx.MultiDiGraph]: The list of dummy MultiDiGraph object.
    """
    graphs = generate_dummy_graphs(num_graphs, num_nodes, degree, edge_weight)
    graphs = add_dummy_cat_node_features(graphs, num_cat, feature_name="act")
    graphs = add_dummy_cat_node_features(graphs, num_cat, feature_name="location")
    graphs = add_dummy_cont_edge_features(graphs, feature_name="duration")
    graphs = add_dummy_cont_edge_features(graphs, feature_name="tst")
    graphs = add_dummy_cont_edge_features(graphs, feature_name="tet")
    graphs = add_dummy_cat_edge_features(graphs, num_cat, feature_name="travel")
    graphs = add_dummy_labels(graphs, num_labels)

    return graphs

@pytest.fixture(scope="session")
def dummy_graphs(
    num_graphs: int = 10,
    num_nodes: int | List[int] = 10,
    degree: int | List[int] = 3,
    edge_weight: int | List[int] = 1,
):
    """
    Generate dummy graphs for testing purposes.

    Args:
        num_graphs (int, optional, default 10): The number of graphs to generate.
        num_nodes (int|List[int], optional, default 10): The number of nodes in each graph.
            If list, the length should be equal to num_graphs, one for each graph.
        degree (int|List[int], optional, default 3): The degree of each node in the graph.
            If list, the length should be equal to num_graphs, one for each graph.
        edge_weight (int|List[int], optional, default 1): The weight of each edge in the graph.
            If list, the length should be equal to num_graphs, one for each graph

    Returns:
        List[nx.MultiDiGraph]: The list of dummy MultiDiGraph object.
    """
    return generate_dummy_graphs(
        num_graphs=num_graphs,
        num_nodes=num_nodes,
        degree=degree,
        edge_weight=edge_weight,
    )

def generate_dummy_graphs(
    num_graphs: int = 10,
    num_nodes: int | List[int] = 10,
    degree: int | List[int] = 3,
    edge_weight: int | List[int] = 1,
) -> List[nx.MultiDiGraph]:
    """
    Generate dummy graphs for testing purposes.

    Args:
        num_graphs (int, optional, default 10): The number of graphs to generate.
        num_nodes (int|List[int], optional, default 10): The number of nodes in each graph.
            If list, the length should be equal to num_graphs, one for each graph.
        degree (int|List[int], optional, default 3): The degree of each node in the graph.
            If list, the length should be equal to num_graphs, one for each graph.
        edge_weight (int|List[int], optional, default 1): The weight of each edge in the graph.
            If list, the length should be equal to num_graphs, one for each graph

    Returns:
        List[nx.MultiDiGraph]: The list of dummy MultiDiGraph object.
    """
    if isinstance(num_nodes, int):
        num_nodes = [num_nodes] * num_graphs
    if isinstance(degree, int):
        degree = [degree] * num_graphs
    if isinstance(edge_weight, int):
        edge_weight = [edge_weight] * num_graphs
    graphs = [
        nx.generators.directed.random_k_out_graph(
            num_nodes[i], degree[i], edge_weight[i]
        )
        for i in range(num_graphs)
    ]

    return graphs

def add_dummy_cont_edge_features(
    graphs: List[nx.MultiDiGraph], feature_name: str = "cont_features"
) -> List[nx.MultiDiGraph]:
    """
    Add dummy continuous features to the edges of the graphs.

    Args:
        graphs (List[nx.MultiDiGraph]): The list of graphs to which features are to be added.
        feature_dim (int): The dimensionality of the features to be added.
        feature_name (str, optional, default "cont_features"): The name of the feature to be added.

    Returns:
        List[nx.MultiDiGraph]: The list of graphs with features added.
    """
    for graph in graphs:
        for edge in graph.edges:
            graph.edges[edge][feature_name] = torch.randn(1).squeeze()
    return graphs

def add_dummy_cont_node_features(
    graphs: List[nx.MultiDiGraph], feature_name: str = "cont_features"
) -> List[nx.MultiDiGraph]:
    """
    Add dummy continuous features to the nodes of the graphs.

    Args:
        graphs (List[nx.MultiDiGraph]): The list of graphs to which features are to be added.
        feature_dim (int): The dimensionality of the features to be added.
        feature_name (str, optional, default "cont_features"): The name of the feature to be added.

    Returns:
        List[nx.MultiDiGraph]: The list of graphs with features added.
    """
    for graph in graphs:
        for node in graph.nodes:
            graph.nodes[node][feature_name] = torch.randn(1).squeeze()
    return graphs

def add_dummy_cat_edge_features(
    graphs: List[nx.MultiDiGraph], num_categories: int = 10, feature_name: str = "cat_features"
) -> List[nx.MultiDiGraph]:
    """
    Add dummy categorical features to the edges of the graphs.

    Args:
        graphs (List[nx.MultiDiGraph]): The list of graphs to which features are to be added.
        num_categories (int, optional, default 10): The number of categories for the features.
        feature_name (str, optional, default "cat_features"): The name of the feature to be added.

    Returns:
        List[nx.MultiDiGraph]: The list of graphs with features added.
    """
    for graph in graphs:
        for edge in graph.edges:
            graph.edges[edge][feature_name] = torch.randint(num_categories, (1,)).squeeze()
    return graphs

def add_dummy_cat_node_features(
    graphs: List[nx.MultiDiGraph], num_categories: int = 10, feature_name: str = "cat_features"
) -> List[nx.MultiDiGraph]:
    """
    Add dummy categorical features to the nodes of the graphs.

    Args:
        graphs (List[nx.MultiDiGraph]): The list of graphs to which features are to be added.
        feature_name (str, optional, default "cat_features"): The name of the feature to be added.

    Returns:
        List[nx.MultiDiGraph]: The list of graphs with features added.
    """
    for graph in graphs:
        for node in graph.nodes:
            graph.nodes[node][feature_name] = torch.randint(num_categories, (1,)).squeeze()
    return graphs

def add_dummy_labels(
    graphs: List[nx.MultiDiGraph], num_classes: int
) -> List[nx.MultiDiGraph]:
    """
    Add dummy labels to the graphs.

    Args:
        graphs (List[nx.MultiDiGraph]): The list of graphs to which labels are to be added.
        num_classes (int): The number of classes for the labels.

    Returns:
        List[nx.MultiDiGraph]: The list of graphs with labels added.
    """
    for graph in graphs:
        graph.graph["labels"] = torch.randint(0, num_classes, (1,))
    return graphs