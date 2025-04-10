from typing import Optional
import pandas as pd
from torch import Tensor


def tokenize(data: pd.Series, encoding_map: Optional[dict] = None) -> Tensor:
    """
    Tokenize a pandas Series into a Tensor. If no encoding map is provided,
    a new encoding map will be created. If an encoding map is provided,
    the function will check if the data matches the existing encoding.
    If not, a UserWarning will be raised.
    Args:
        data (pd.Series): input Series to tokenize.
        encoding_map (dict, optional): existing encodings to use. Defaults to None.
    Returns:
        Tensor: Tensor of encodings and encoding map.
    """
    if encoding_map:
        missing = set(data.unique()) - set(encoding_map.keys())
        if missing:
            raise UserWarning(
                f"""
                Categories in data do not match existing categories.
                {missing} not found.
                Please specify the new categories in the encoding.
                Your existing encodings are: {encoding_map}.
"""
            )
        encoded = pd.Categorical(data, categories=encoding_map.keys())
    else:
        encoded = pd.Categorical(data)
        encoding_map = {e: i for i, e in enumerate(encoded.categories)}
    encoded = Tensor(encoded.codes.copy()).int()
    return encoded, encoding_map
