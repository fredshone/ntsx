from ntsx.utils import split_dataset
from ntsx.nx_to_torch import nx_to_torch_geo

def test_split_dataset(dummy_graphs):

    #generate a dummy dataset
    graphs = dummy_graphs
    dataset = nx_to_torch_geo(graphs)

    # split the dataset into training and testing sets
    train_set, test_set = split_dataset(dataset, 0.8)

    # check the lengths of the resulting sets
    assert len(train_set) + len(test_set) == len(dataset), "Train and test sets do not sum up to the original dataset length"
    assert len(train_set) == int(0.8 * len(dataset)), "Train set length is not 80% of the original dataset length"
    assert len(test_set) == int(0.2 * len(dataset)), "Test set length is not 20% of the original dataset length"

    assert len(train_set) > 0, "Train set is empty"
    assert len(test_set) > 0, "Test set is empty"

def test_split_dataset_empty():
    # create an empty dataset
    dataset = []

    # split the dataset into training and testing sets
    train_set, test_set = split_dataset(dataset, 0.8)

    # check the lengths of the resulting sets
    assert len(train_set) == 0, "Train set is not empty"
    assert len(test_set) == 0, "Test set is not empty"