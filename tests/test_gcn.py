from ntsx.utils import (
    add_dummy_features,
    add_dummy_labels,
    generate_dummy_graphs,
    split_dataset,
    train_epoch,
    eval_model,
)
from ntsx.data_loader import build_loader, nx_to_torch_geo
from ntsx.gcn import GCN

import torch
import pytest
import warnings

torch.manual_seed(0)


@pytest.mark.parametrize(
    "num_graphs, num_nodes, degree, edge_weight, hidden_channels, n_in, n_out, batch_size, shuffle, lr, device, dropout",
    [
        (10, 5, 3, 1, 16, 4, 2, 2, True, 0.01, "cpu", 0.5),
        (10, 5, 3, 1, 16, 4, 2, 2, False, 0.01, "cpu", 0.5),
        (10, 5, 3, 1, 16, 4, 2, 2, True, 0.01, "cuda:0", 0.5),
        (10, 5, 3, 1, 16, 4, 2, 2, False, 0.01, "cuda:0", 0.5),
        (10, 5, 3, 1, 16, 4, 2, 2, True, 0.01, "cpu", 0),
        (10, 5, 3, 1, 16, 4, 2, 2, False, 0.01, "cpu", 0),
        (10, 5, 3, 1, 16, 4, 2, 2, True, 0.01, "cuda:0", 0),
        (10, 5, 3, 1, 16, 4, 2, 2, False, 0.01, "cuda:0", 0),
    ],
)
def test_gcn(
    num_graphs,
    num_nodes,
    degree,
    edge_weight,
    hidden_channels,
    n_in,
    n_out,
    batch_size,
    shuffle,
    lr,
    device,
    dropout,
):
    # Load dummy graphs
    graphs = generate_dummy_graphs(num_graphs, num_nodes, degree, edge_weight)
    graphs = add_dummy_features(graphs, n_in)
    graphs = add_dummy_labels(graphs, n_out)

    dataset = nx_to_torch_geo(graphs)
    dataset_train, dataset_test = split_dataset(dataset, 0.8)

    train_loader = build_loader(dataset_train, batch_size=batch_size, shuffle=shuffle)
    test_loader = build_loader(dataset_test, batch_size=batch_size, shuffle=shuffle)

    model = GCN(hidden_channels, n_in, n_out, dropout=dropout).to(torch.device(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, torch.device(device)
        )

        assert train_loss.shape == torch.Size([]), (
            f"Expected torch.Size([]), but got {train_loss.shape}"
        )
        assert train_loss.dtype == torch.float32, (
            f"Expected torch.float32, but got {train_loss.dtype}"
        )
        assert train_loss.device == torch.device(device), (
            f"Expected device {device}, but got {train_loss.device}"
        )
        assert train_loss >= 0, f"Expected non-negative loss, but got {train_loss}"
        if train_loss > 1:
            warnings.warn(
                "The train loss is high. The model may not be training properly."
            )

        eval_loss = eval_model(model, test_loader, torch.device(device))

        assert eval_loss.shape == torch.Size([]), (
            f"Expected torch.Size([]), but got {eval_loss.shape}"
        )
        assert eval_loss.dtype == torch.float32, (
            f"Expected torch.float32, but got {eval_loss.dtype}"
        )
        assert eval_loss.device == torch.device(device), (
            f"Expected device {device}, but got {eval_loss.device}"
        )
        assert eval_loss >= torch.tensor(
            [0], device=torch.device(device)
        ), f"Expected non-negative loss, but got {eval_loss}"
        if eval_loss > 1:
            warnings.warn(
                "The eval loss is high. The model may not be evaluating properly."
            )

