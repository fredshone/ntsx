import torch.nn as nn
from torch import stack

class MultiTokenEmbedSum(nn.Module):
    def __init__(self, label_embed_sizes: list[int], hidden_size: int = 32):
        """Embed tokens and add them together."""
        super(MultiTokenEmbedSum, self).__init__()
        self.embeds = nn.ModuleList(
            [nn.Embedding(s, hidden_size) for s in label_embed_sizes]
        )

    def forward(self, x):
        return stack([embed(x[i]) for i, embed in enumerate(self.embeds)], dim=-1).sum(
            dim=-1
        )