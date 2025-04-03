from torch import Tensor, stack
from typing import List, Optional
import pandas as pd

from ntsx.encoders.utils import tokenize


class TableTokeniser:

    def __init__(
        self,
        data: pd.DataFrame,
        include: Optional[list] = None,
        exclude: Optional[list] = None,
    ):
        """Tokenise a pandas dataframe into a Tensor,
        and initialise mapping for further encoding and decoding.
        Args:
            data (pd.DataFrame): input dataframe to tokenise.
            include (list, optional): columns to include. Defaults to None.
            exclude (list, optional): columns to exclude. Defaults to None.
        """
        columns = data.columns.tolist()
        columns = [col for col in columns if col not in ["pid", "iid", "hid"]]
        if include is not None:
            columns = [col for col in columns if col in include]
        if exclude is not None:
            columns = [col for col in columns if col not in exclude]

        if not columns:
            raise UserWarning("No columns found to encode in table.")

        self.columns = columns
        self.configure(data)

        print(
            f"""{self} configured with columns: {self.columns}
            embed_types: {self.embed_types}
            embed_sizes: {self.embed_sizes}
            maps: {self.maps}
            dtypes: {self.dtypes}"""
        )

    def configure(self, data: pd.DataFrame) -> None:
        """Configure the tokeniser by encoding the dataframe columns.
        Args:
            data (pd.DataFrame): input dataframe to configure.
        """

        n = len(self.columns)
        self.embed_types = [None for _ in range(n)]
        self.embed_sizes = [None for _ in range(n)]
        self.maps = [None for _ in range(n)]
        self.dtypes = [None for _ in range(n)]

        for i, column in enumerate(self.columns):
            if column not in data.columns:
                raise UserWarning(f"Column '{column}' not found in attributes")

            # todo: Assume all nominal categorical encoding for now
            _, map = tokenize(data[column], None)
            self.embed_types[i] = "categorical"
            self.embed_sizes[i] = len(map)
            self.maps[i] = map
            self.dtypes[i] = data[column].dtype

    def encode(self, data: pd.DataFrame) -> Tensor:
        """Encode the dataframe into a Tensor.
        Args:
            data (pd.DataFrame): input dataframe to encode.
        Returns:
            Tensor: encoded dataframe.
        """
        encoded = []
        for column, map in zip(self.columns, self.maps):
            if column not in data.columns:
                raise UserWarning(f"Column '{column}' not found in attributes")
            column_encoded, _ = tokenize(data[column], map)
            encoded.append(column_encoded)

        if not encoded:
            raise UserWarning("No attribute encoding found.")

        # todo: add loss weightings in future

        encoded = stack(encoded, dim=-1).long()
        return encoded

    def decode(self, data: List[Tensor]) -> pd.DataFrame:
        """Decode Tensor of tokens back into dataframe.

        Args:
            data (List[Tensor]): input Tensor of tokens to decode.

        Returns:
            pd.DataFrame: decoded dataframe.
        """
        decoded = {"pid": list(range(data.shape[0]))}
        for i, (column, map, dtype) in enumerate(
            zip(self.columns, self.maps, self.dtypes)
        ):
            encoding = {i: name for name, i in map.items()}
            tokens = data[:, i].tolist()
            decoded[column] = pd.Series([encoding[i] for i in tokens]).astype(dtype)
        return pd.DataFrame(decoded)

    def argmax_decode(self, data: List[Tensor]) -> pd.DataFrame:
        argmaxed = [d.argmax(dim=-1) for d in data]
        return self.decode(argmaxed)

    def multinomial_decode(self, data: List[Tensor]) -> pd.DataFrame:
        sampled = [d.multinomial(1).squeeze() for d in data]
        return self.decode(sampled)
