import torch
import networkx as nx
from typing import List


def generate_dummy_graphs(
    num_graphs: int,
    num_nodes: int | List[int] = 10,
    degree: int | List[int] = 3,
    edge_weight: int | List[int] = 1,
) -> List[nx.MultiDiGraph]:
    """
    Generate dummy graphs for testing purposes.

    Args:
        num_graphs (int): The number of graphs to generate.
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


def add_dummy_features(
    graphs: List[nx.MultiDiGraph], feature_dim: int
) -> List[nx.MultiDiGraph]:
    """
    Add dummy features to the nodes and edges of the graphs.

    Args:
        graphs (List[nx.MultiDiGraph]): The list of graphs to which features are to be added.
        feature_dim (int): The dimensionality of the features to be added.

    Returns:
        List[nx.MultiDiGraph]: The list of graphs with features added.
    """
    for graph in graphs:
        for node in graph.nodes:
            graph.nodes[node]["features"] = torch.randn(feature_dim)
        for edge in graph.edges:
            graph.edges[edge]["features"] = torch.randn(feature_dim)
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


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train the model on the training data for one epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): The DataLoader object containing the training data.
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        criterion (torch.nn.Module): The loss function to be used.
        device (str): The device to run the training on.

    Returns:
        float: The average loss over the training data.
    """
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.features, data.edge_index, data.batch)
        loss = criterion(output, data.labels)
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss / len(train_loader)


def eval_model(model, loader, device):
    """
    Evaluate the model on the validation or test data.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        loader (DataLoader): The DataLoader object containing the validation or test data.
        device (str): The device to run the evaluation on.

    Returns:
        float: The accuracy of the model on the validation or test data.
    """
    model.eval()
    total_loss = 0
    for data in loader:
        data.to(device)
        with torch.no_grad():
            out = model(data.features, data.edge_index, data.batch)
            loss = torch.nn.functional.cross_entropy(out, data.labels)
            total_loss += loss
    return total_loss / len(loader.dataset)


def split_dataset(graphs: List[nx.MultiDiGraph], train_ratio: float = 0.8):
    """
    Split the dataset into training and validation sets.

    Args:
        graphs (List[nx.MultiDiGraph]): The list of graphs to be split.
        train_ratio (float, optional, default 0.8): The ratio of the training set.

    Returns:
        List[nx.MultiDiGraph]: The training set.
        List[nx.MultiDiGraph]: The validation set.
    """

    return torch.utils.data.random_split(graphs, [train_ratio, 1 - train_ratio])
