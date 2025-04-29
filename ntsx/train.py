from pathlib import Path

from torch_geometric.loader import DataLoader
from ntsx.nx_to_torch import nx_to_torch_geo
import torch
import torch.nn.functional as F

from ntsx import graph_ops, nts_to_nx
from ntsx import read_nts
from ntsx.utils import split_dataset, train_epoch, eval_model
from ntsx.encoders.trip_encoder import TripEncoder
from ntsx.encoders.table_encoder import TableTokeniser
from ntsx.models.models import GCNGraphLabeller, GATGraphLabeller


def main(model_name="gcn"):
    """
    Main function to run the model.
    Args:
        model_name (str): The name of the model to be used. Options are "gcn" or "gat".
    """
    # load data (synthesised from UK NTS)

    dir = Path("data/NTS/")
    trips_path = dir / "trips.tab"
    attributes_path = dir / "individuals.tab"
    hhs_path = dir / "households.tab"

    years = [2021]

    write_dir = Path("tmp")
    write_dir.mkdir(exist_ok=True)

    # load data from disk
    trips, labels = read_nts.load_nts(trips_path, attributes_path, hhs_path, years=years)

    # assign human readable values to the labels
    labels = read_nts.label_mapping(labels)

    # initaite the encoders
    label_encoder = TableTokeniser(labels, verbose=False)
    trip_encoder = TripEncoder(trips)

    # first encode the trips and labels tables
    trips_encoded = trip_encoder.encode_trips_table(trips)
    labels_encoded = label_encoder.encode_table(labels)

    # then build individuals and then days graphs from the trips table, note that we only merge on home (2)
    individuals = nts_to_nx.to_individuals_nx(trips_encoded, attribute_data=labels_encoded)
    days = []
    for ind in individuals:
        g = graph_ops.anchor_activities(ind, [2])
        g = graph_ops.merge_similar(g, duration_tolerance=0.2)

        # now we can create a graph for each day
        indiv_days = [d for _, d in graph_ops.iter_days(g, stop=None)]
        days.extend(indiv_days)

    # now we can create a graph dataset
    dataset = nx_to_torch_geo(days)

    # and split into train and test sets
    dataset_train, dataset_test = split_dataset(dataset, train_ratio=0.8)
    dataset_train, dataset_val = split_dataset(dataset_train, train_ratio=0.8)

    # finally we can create a dataloader
    loader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=64, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node_embed_sizes = [
        trip_encoder.embed_sizes()["oact"],
        trip_encoder.embed_sizes()["ozone"],
    ]
    target_size = label_encoder.embed_sizes()["work_status"]

    if model_name == "gcn":
        model = GCNGraphLabeller(
            node_embed_sizes=node_embed_sizes, target_size=target_size, hidden_size=32
        ).to(device)
    elif model_name == "gat":
        edge_embed_sizes = [
            trip_encoder.embed_sizes()["duration"],
            trip_encoder.embed_sizes()["tst"],
            trip_encoder.embed_sizes()["tet"],
        ]
        model = GATGraphLabeller(
            node_embed_sizes=node_embed_sizes,
            edge_embed_sizes=edge_embed_sizes,
            target_size=target_size,
            hidden_size=32,
        ).to(device)
    else:
        raise ValueError("Model name must be either 'gcn' or 'gat'")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)
    criterion = torch.nn.BCELoss()

    model.train()
    print(f"Training {model_name} model")
    for epoch in range(10):
        train_loss = train_epoch(
            model,
            loader_train,
            optimizer,
            criterion,
            device,
            label_name="work_status"
        )
        val_loss = eval_model(
            model,
            loader_val,
            device,
            label_name="work_status"
        )
        print(
            f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )
    
    # evaluate the model on the test set
    test_loss = eval_model(
        model,
        loader_test,
        device,
        label_name="work_status"
    )
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main(model_name="gcn")
    main(model_name="gat")
