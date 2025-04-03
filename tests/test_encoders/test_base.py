import pytest

from pandas import Series

from ntsx.encoders.base_encoders import (
    ContinuousEncoder,
    CategoricalTokeniser,
    TimeEncoder,
)
from pandas.testing import assert_series_equal


def test_cont_encoder_encode_int():
    x = Series([1, 2, 3])
    encoder = ContinuousEncoder(x)
    encoded = encoder.encode(x)
    assert_series_equal(encoded, Series([0.0, 0.5, 1.0]))
    decoded = encoder.decode(encoded)
    assert_series_equal(decoded, x)


def test_cont_encoder_encode_float():
    x = Series([1.0, 2.0, 3.0])
    encoder = ContinuousEncoder(x)
    encoded = encoder.encode(x)
    assert_series_equal(encoded, Series([0.0, 0.5, 1.0]))
    decoded = encoder.decode(encoded)
    assert_series_equal(decoded, x)


def test_cont_encoder_encode_obj():
    x = Series(["A", "B", "C"])
    with pytest.raises(UserWarning) as w:
        _ = ContinuousEncoder(x)
        assert w.contains("ContinuousEncoder only supports float and int data types.")


def test_time_encoder_encode_int():
    x = Series([0, 1440, 2880])
    encoder = TimeEncoder(x, min_value=0, day_value=1440)
    encoded = encoder.encode(x)
    assert_series_equal(encoded, Series([0.0, 1.0, 2.0]))
    decoded = encoder.decode(encoded)
    assert_series_equal(decoded, x)


def test_time_encoder_encode_float():
    x = Series([0.0, 1440.0, 2880.0])
    encoder = TimeEncoder(x, min_value=0, day_value=1440)
    encoded = encoder.encode(x)
    assert_series_equal(encoded, Series([0.0, 1.0, 2.0]))
    decoded = encoder.decode(encoded)
    assert_series_equal(decoded, x)


def test_time_encoder_encode_obj():
    x = Series(["A", "B", "C"])
    with pytest.raises(UserWarning) as w:
        _ = TimeEncoder(x, min_value=0, day_value=1440)
        assert w.contains("TimeEncoder only supports float and int data types.")


def test_cont_minmax_encoder_encode_int():
    x = Series([1, 2, 3])
    encoder = CategoricalTokeniser(x)
    encoded = encoder.encode(x)
    assert_series_equal(encoded, Series([0, 1, 2]))
    decoded = encoder.decode(encoded)
    assert_series_equal(decoded, x)


def test_cont_minmax_encoder_encode_float():
    x = Series([1.0, 2.0, 3.0])
    with pytest.raises(UserWarning) as w:
        _ = CategoricalTokeniser(x)
        assert w.contains("CategoricalEncoder only supports object and int data types.")


def test_cont_minmax_encoder_encode_obj():
    x = Series(["A", "B", "C"])
    encoder = CategoricalTokeniser(x)
    encoded = encoder.encode(x)
    assert_series_equal(encoded, Series([0, 1, 2]))
    decoded = encoder.decode(encoded)
    assert_series_equal(decoded, x)
