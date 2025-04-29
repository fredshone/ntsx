from torch import nn, cat, stack
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool

from ntsx.models.embed import MultiTokenEmbedSum


class GCNGraphLabeller(nn.Module):
    def __init__(
        self,
        node_embed_sizes: list[int],
        target_size: int,
        hidden_size: int = 32,
        dropout: float = 0.5,
    ):
        """A simple GNN model for graph classification."""
        super().__init__()
        self.node_embed = MultiTokenEmbedSum(node_embed_sizes, hidden_size)
        self.conv1 = GCNConv(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, target_size)
        self.dropout = dropout

    def forward(self, data):
        x = [data.act, data.location]
        edge_index, batch = data.edge_index, data.batch
        x = self.node_embed(x)
        x = F.relu(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class GATGraphLabeller(nn.Module):
    def __init__(
        self,
        node_embed_sizes: list[int],
        edge_embed_sizes: list[int],
        target_size: int,
        hidden_size: int = 32,
        dropout: float = 0.5,
    ):
        """A simple GAT model for graph classification with edge and node attributes."""
        super().__init__()
        self.node_embed = MultiTokenEmbedSum(node_embed_sizes, hidden_size)
        self.edge_embed = MultiTokenEmbedSum(edge_embed_sizes, hidden_size)
        self.conv1 = GATConv(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, target_size)
        self.dropout = dropout

    def forward(self, data):
        x = [data.act, data.location]
        x = self.node_embed(x)

        x_edge_cont = stack([data.duration, data.tst, data.tet], dim=1)
        x_edge_cat = [data.travel]
        x_edge_cat = self.edge_embed(x_edge_cat)
        x_edge = cat([x_edge_cat, x_edge_cont], dim=-1)

        edge_index, batch = data.edge_index, data.batch

        x = F.relu(x)
        x = self.conv1(x, edge_index, x_edge)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
