import pytest

from pandas import Series
from torch import Tensor
from torch.testing import assert_close

from ntsx.encoders.utils import tokenize


def test_fresh_tokenise():
    data = Series(["M", "F", "F"])
    tokens, mapping = tokenize(data)
    assert_close(tokens, Tensor([1, 0, 0]).int())
    assert mapping == {"F": 0, "M": 1}


def test_empty_tokenise():
    data = Series([])
    tokens, mapping = tokenize(data)
    assert_close(tokens, Tensor([]).int())
    assert mapping == {}


def test_re_tokenise():
    data = Series(["M", "F", "F"])
    mapping = {"F": 0, "M": 1}
    tokens, mapping = tokenize(data, mapping)
    assert_close(tokens, Tensor([1, 0, 0]).int())
    assert mapping == {"F": 0, "M": 1}


def test_re_tokenise_bad_mapping():
    data = Series(["M", "F", "F"])
    bad_mapping = {"F": 0, "X": 1}
    with pytest.raises(UserWarning) as w:
        tokens, mapping = tokenize(data, bad_mapping)
        assert w.contains("Categories in data do not match existing categories.")
        assert w.contains("X")
