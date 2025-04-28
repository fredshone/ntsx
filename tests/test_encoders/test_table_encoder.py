import pytest

from pandas import DataFrame
from torch import Tensor
from torch.testing import assert_close

from ntsx.encoders.table_encoder import TableTokeniser


def test_encoder_all():
    data = DataFrame({"pid": [0, 1, 2], "A": [34, 96, 15], "B": ["M", "F", "F"]})
    encoder = TableTokeniser(data)
    encoded = encoder.encode(data)
    expected = Tensor([[1, 1], [2, 0], [0, 0]]).long()
    assert_close(encoded, expected)
    assert encoder.columns == ["A", "B"]
    assert encoder.embed_types() == {"A": "categorical", "B": "categorical"}
    assert encoder.embed_sizes() == {"A": 3, "B": 2}


def test_encoder_include():
    data = DataFrame({"pid": [0, 1, 2], "A": [34, 96, 15], "B": ["M", "F", "F"]})
    encoder = TableTokeniser(data, include=["A"])
    encoded = encoder.encode(data)
    expected = Tensor([[1], [2], [0]]).long()
    assert_close(encoded, expected)
    assert encoder.columns == ["A"]
    assert encoder.embed_types()["A"] == "categorical"
    assert encoder.embed_sizes()["A"] == 3


def test_encoder_exclude():
    data = DataFrame({"pid": [0, 1, 2], "A": [34, 96, 15], "B": ["M", "F", "F"]})
    encoder = TableTokeniser(data, exclude=["A"])
    encoded = encoder.encode(data)
    expected = Tensor([[1], [0], [0]]).long()
    assert_close(encoded, expected)
    assert encoder.columns == ["B"]
    assert encoder.embed_types()["B"] == "categorical"
    assert encoder.embed_sizes()["B"] == 2


def test_encoder_missing_include():
    data = DataFrame({"pid": [0, 1, 2], "A": [34, 96, 15], "B": ["M", "F", "F"]})
    with pytest.raises(UserWarning) as w:
        _ = TableTokeniser(data, include=["C"])
        assert w.message.contains("No columns found to encode in table.")


def test_encoder_missing_data():
    data = DataFrame({"pid": [0, 1, 2], "hid": [0, 0, 0]})
    with pytest.raises(UserWarning) as w:
        _ = TableTokeniser(data)
        assert w.message.contains("No columns found to encode in table.")


def test_encode_decode():
    data = DataFrame({"pid": [0, 1, 2], "A": [34, 96, 15], "B": ["M", "F", "F"]})
    encoder = TableTokeniser(data)
    encoded = encoder.encode(data)
    decoded = encoder.decode(encoded)
    assert decoded.equals(data)


def test_encode_unknown_token():
    data = DataFrame({"pid": [0, 1, 2], "A": [34, 96, 15], "B": ["M", "F", "F"]})
    encoder = TableTokeniser(data)
    bad_data = DataFrame({"pid": [3, 4, 5], "A": [96, 34, 15], "B": ["M", "F", "X"]})
    with pytest.raises(UserWarning) as w:
        _ = encoder.encode(bad_data)
        assert w.message.contains(
            "Categories in data do not match existing categories."
        )
        assert w.message.contains("X")


def test_decode_unknown_token():
    data = DataFrame({"pid": [0, 1, 2], "A": [34, 96, 15], "B": ["M", "F", "F"]})
    encoder = TableTokeniser(data)
    encoded = Tensor([[1, 1], [2, 0], [0, 2]]).long()
    with pytest.raises(UserWarning) as w:
        _ = encoder.decode(encoded)
        assert w.message.contains("2")
