from torch import Tensor, stack
from typing import List, Optional
import pandas as pd

from ntsx.encoders.base_encoders import CategoricalTokeniser


class TableTokeniser:

    def __init__(
        self,
        data: pd.DataFrame,
        include: Optional[list] = None,
        exclude: Optional[list] = None,
        verbose: bool = False,
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
        self.configure(data, verbose=verbose)

    def configure(self, data: pd.DataFrame, verbose: bool = False) -> None:
        """Configure the tokeniser by encoding the dataframe columns.
        Args:
            data (pd.DataFrame): input dataframe to configure.
            verbose (bool, optional): print the configuration. Defaults to False.
        """

        self.encoders = []

        for column in self.columns:
            if column not in data.columns:
                raise UserWarning(f"Column '{column}' not found in attributes")

            # todo: Assume all nominal categorical encoding for now
            # todo: later can add a config and other types
            self.encoders.append(
                CategoricalTokeniser(data[column], column, verbose=verbose)
            )

    def encode(self, data: pd.DataFrame) -> Tensor:
        """Encode the dataframe into a Tensor.
        Args:
            data (pd.DataFrame): input dataframe to encode.
        Returns:
            Tensor: encoded dataframe.
        """
        encoded = []
        for column, encoder in zip(self.columns, self.encoders):
            if column not in data.columns:
                raise UserWarning(f"Column '{column}' not found in attributes")
            column_encoded = encoder.encode(data[column])
            encoded.append(column_encoded)

        if not encoded:
            raise UserWarning("No attribute encoding found.")

        # todo: add loss weightings in future

        encoded = stack(encoded, dim=-1).long()
        return encoded

    def embed_types(self) -> List[str]:
        """Get the types of the embeddings.
        Returns:
            List[str]: list of types of the embeddings.
        """
        return [encoder.encoding for encoder in self.encoders]

    def embed_sizes(self) -> List[int]:
        """Get the sizes of the embeddings.
        Returns:
            List[int]: list of sizes of the embeddings.
        """
        return [encoder.size for encoder in self.encoders]

    def decode(self, data: List[Tensor]) -> pd.DataFrame:
        """Decode Tensor of tokens back into dataframe.

        Args:
            data (List[Tensor]): input Tensor of tokens to decode.

        Returns:
            pd.DataFrame: decoded dataframe.
        """
        decoded = {"pid": list(range(data.shape[0]))}
        for i, (column, encoder) in enumerate(zip(self.columns, self.encoders)):
            tokens = data[:, i].tolist()
            decoded[column] = encoder.decode(tokens)
        return pd.DataFrame(decoded)

    def argmax_decode(self, data: List[Tensor]) -> pd.DataFrame:
        argmaxed = [d.argmax(dim=-1) for d in data]
        return self.decode(argmaxed)

    def multinomial_decode(self, data: List[Tensor]) -> pd.DataFrame:
        sampled = [d.multinomial(1).squeeze() for d in data]
        return self.decode(sampled)
