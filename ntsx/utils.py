import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader
from typing import List, Tuple
from networkx import MultiDiGraph


def train_epoch(
    model: Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: Module,
    device: torch.device,
    label_name: str = "labels",
) -> float:
    """
    Train the model on the training data for one epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): The DataLoader object containing the training data.
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        criterion (torch.nn.Module): The loss function to be used.
        device (torch.device): The device to run the training on.
        label_name (str, optional, default "labels"): The name of the label in the data.

    Returns:
        float: The average loss over the training data.
    """
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        output = model(data)
        loss = criterion(output, data[label_name])
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss / len(train_loader)


def eval_model(
    model: Module,
    loader: DataLoader,
    device: torch.device,
    label_name: str = "labels",
) -> float:
    """
    Evaluate the model on the validation or test data.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        loader (DataLoader): The DataLoader object containing the validation or test data.
        device (torch.device): The device to run the evaluation on.
        label_name (str, optional, default "labels"): The name of the label in the data.

    Returns:
        float: The accuracy of the model on the validation or test data.
    """
    model.eval()
    total_loss = 0
    for data in loader:
        data.to(device)
        with torch.no_grad():
            out = model(data)
            loss = torch.nn.functional.nll_loss(out, data[label_name])
            total_loss += loss
    return total_loss / len(loader)


def split_dataset(
    graphs: List[MultiDiGraph], train_ratio: float = 0.8
) -> Tuple[List[MultiDiGraph], List[MultiDiGraph]]:
    """
    Split the dataset into training and validation sets.

    Args:
        graphs (List[nx.MultiDiGraph]): The list of graphs to be split.
        train_ratio (float, optional, default 0.8): The ratio of the training set.

    Returns:
        List[nx.MultiDiGraph]: The training set.
        List[nx.MultiDiGraph]: The validation set.
    """
    len_train = int(len(graphs) * train_ratio)
    len_test = len(graphs) - len_train

    return torch.utils.data.random_split(graphs, [len_train, len_test])
