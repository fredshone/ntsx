import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import List, Optional

def nx_to_torch_geo(graphs: List[nx.MultiDiGraph]) -> List[Data]:
    """
    Convert a list of NetworkX MultiDiGraph objects to a list of PyTorch Geometric Data objects.

    Args:
        graphs (List[nx.MultiDiGraph]): The list of NetworkX MultiDiGraph objects.

    Returns:
        List[Data]: The list of PyTorch Geometric Data objects.
    """
    return [from_networkx(graph) for graph in graphs]


def build_loader(
    graphs: List[Data],
    batch_size: int = 1,
    shuffle: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Build a PyTorch Geometric DataLoader object from a list of PyTorch Geometric Data objects.

    Args:
        graphs (List[Data]): The list of PyTorch Geometric Data objects.
        batch_size (int, optional, default 1): The size of each batch.
        shuffle (bool, optional, default True): Whether to shuffle the data.
        **kwargs: Additional keyword arguments from torch.utils.data.DataLoader 
            or torch_geometric.loader.DataLoader to pass to the DataLoader.

    Returns:
        DataLoader: The PyTorch Geometric DataLoader object.
    """
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle, **kwargs)
