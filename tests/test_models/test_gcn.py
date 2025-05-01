from ntsx.utils import (
    split_dataset,
    train_epoch,
    eval_model,
)
from ntsx.nx_to_torch import build_loader, nx_to_torch_geo
from ntsx.models.models import GCNGraphLabeller

import torch
import pytest
import warnings
from torch_geometric.loader import DataLoader

torch.manual_seed(0)

if torch.cuda.is_available():
    CUDA_AVAILABLE = True
else:
    CUDA_AVAILABLE = False


@pytest.mark.parametrize(
    "shuffle, device, dropout",
    [
        (True, "cpu", 0.5),
        (False, "cpu", 0.5),
        (True, "cuda:0", 0.5),
        (False, "cuda:0", 0.5),
        (True, "cpu", 0),
        (False, "cpu", 0),
        (True, "cuda:0", 0),
        (False, "cuda:0", 0),
    ],
)
def test_gcn(
    shuffle, device, dropout, dummy_graphs_for_gcn,
):
    # skip test if CUDA is not available
    if device == "cuda:0" and not CUDA_AVAILABLE:
        pytest.skip("CUDA is not available. Skipping tests that require CUDA.")

    # load dummy graphs
    graphs = dummy_graphs_for_gcn

    assert len(graphs) == 10, f"Expected 10 graphs, but got {len(graphs)}"
    assert (
        len(graphs[0].nodes) == 10
    ), f"Expected 10 nodes, but got {len(graphs[0].nodes)}"
    assert (
        len(graphs[0].edges) == 10 * 3
    ), f"Expected 30 edges, but got {len(graphs[0].edges)}"

    # turn graphs into torch_geometric data objects and split into train and test sets
    dataset = nx_to_torch_geo(graphs)
    dataset_train, dataset_test = split_dataset(dataset, 0.8)

    # check split dataset
    print(len(dataset), len(dataset_train), len(dataset_test))
    assert (
        len(dataset_train) == 8
    ), f"Expected 8 graphs, but got {len(dataset_train)}"
    assert (
        len(dataset_test) == 2
    ), f"Expected 2 graphs, but got {len(dataset_test)}"

    # build data loaders
    train_loader = build_loader(dataset_train, batch_size=2, shuffle=shuffle)
    test_loader = build_loader(dataset_test, batch_size=2, shuffle=shuffle)

    # check if the data loaders are empty
    assert len(train_loader) > 0, "Train loader is empty."
    assert len(test_loader) > 0, "Test loader is empty."
    assert (
        type(train_loader) == DataLoader
    ), f"Expected DataLoader, but got {type(train_loader)}"
    assert (
        type(test_loader) == DataLoader
    ), f"Expected DataLoader, but got {type(test_loader)}"

    # instantiate model, optimizer, and loss function
    node_embed_sizes = [10]
    model = GCNGraphLabeller(
        node_embed_sizes,
        target_size=3,
        hidden_size=16,
        dropout=dropout,
    ).to(torch.device(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # train and evaluate model for 1 epoch
    for _ in range(1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, torch.device(device)
        )

        # train tests
        assert train_loss.shape == torch.Size(
            []
        ), f"Expected torch.Size([]), but got {train_loss.shape}"
        assert (
            train_loss.dtype == torch.float32
        ), f"Expected torch.float32, but got {train_loss.dtype}"
        assert train_loss.device == torch.device(
            device
        ), f"Expected device {device}, but got {train_loss.device}"
        assert train_loss >= 0, f"Expected non-negative loss, but got {train_loss}"
        if train_loss > 2:
            warnings.warn(
                "The train loss is high. The model may not be training properly. It could also be randomness."
            )

        eval_loss = eval_model(model, test_loader, torch.device(device))

        # eval tests
        assert eval_loss.shape == torch.Size(
            []
        ), f"Expected torch.Size([]), but got {eval_loss.shape}"
        assert (
            eval_loss.dtype == torch.float32
        ), f"Expected torch.float32, but got {eval_loss.dtype}"
        assert eval_loss.device == torch.device(
            device
        ), f"Expected device {device}, but got {eval_loss.device}"
        assert eval_loss >= torch.tensor(
            [0], device=torch.device(device)
        ), f"Expected non-negative loss, but got {eval_loss}"
        if eval_loss > 2:
            warnings.warn(
                "The eval loss is high. The model may not be evaluating properly. It could also be randomness."
            )
