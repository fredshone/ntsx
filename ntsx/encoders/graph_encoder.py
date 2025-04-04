from pandas import DataFrame
from typing import Iterable, Any

from ntsx.encoders.base_encoders import (
    CategoricalTokeniser,
    TimeEncoder,
)


class GraphEncoder:

    def __init__(self, trips: DataFrame):
        """
        GraphEncoder is used to encode the graph data.

        Todo:
        - can only be initialised with a trips table.

        Beware that trip column names are hardcoded.

        Args:
            trips (DataFrame): input data to be encoded
        """
        self.encoders = {
            "mode": CategoricalTokeniser(trips["mode"]),
            "oact": CategoricalTokeniser(trips["oact"]),
            "dact": CategoricalTokeniser(trips["dact"]),
            "day": CategoricalTokeniser(trips["day"]),
            "tst": TimeEncoder(trips["tst"], min_value=0, day_value=1440),
            "tet": TimeEncoder(trips["tet"], min_value=0, day_value=1440),
            "ozone": CategoricalTokeniser(trips["ozone"]),
            "dzone": CategoricalTokeniser(trips["dzone"]),
        }

    def get_encoding(self, name: str) -> str:
        """Get the encoder for the given name."""
        if name not in self.encoders:
            raise UserWarning(f"Encoder for {name} not found.")
        return self.encoders[name].encoding

    def get_size(self, name: str) -> int:
        """Get the size of the encoder for the given name."""
        if name not in self.encoders:
            raise UserWarning(f"Encoder for {name} not found.")
        return self.encoders[name].size

    def encode_trips_table(self, trips: DataFrame) -> DataFrame:
        """Encode the trips table using the encoders.
        Args:
            trips (DataFrame): input data to be encoded
        Returns:
            DataFrame: encoded data
        """
        encoded = trips.copy()
        for name, encoder in self.encoders.items():
            encoded[name] = encoder.encode(trips[name])
        return encoded

    def encode(self, feature, name: str) -> Any:
        """Encode the feature using the named encoder.
        Args:
            feature (Any): input data to be encoded
            name (str): name of the feature to be encoded
        Returns:
            Any: encoded data
        """
        if name not in self.encoders:
            raise UserWarning(f"Encoder for {name} not found.")
        return self.encoders[name].encode([feature]).tolist()[0]

    def decode(self, feature: Any, name: str) -> Any:
        """
        Decode the feature using the named encoder.
        Args:
            feature (Any): input data to be decoded
            name (str): name of the feature to be decoded
        Returns:
            Any: decoded data
        """
        if name not in self.encoders:
            raise UserWarning(f"Encoder for {name} not found.")
        return self.encoders[name].decode([feature]).tolist()[0]

    def encode_iterable(self, features: Iterable, name: str) -> Any:
        """Encode the feature using the named encoder.
        Args:
            features (Iterable): input data to be encoded
            name (str): name of the feature to be encoded
        Returns:
            Iterable: encoded data
        """
        if name not in self.encoders:
            raise UserWarning(f"Encoder for {name} not found.")
        return self.encoders[name].encode(features)

    def decode_iterable(self, features: Iterable, name: str) -> Any:
        """
        Decode the feature using the named encoder.
        Args:
            features (Iterable): input data to be decoded
            name (str): name of the feature to be decoded
        Returns:
            Iterable: decoded data
        """
        if name not in self.encoders:
            raise UserWarning(f"Encoder for {name} not found.")
        return self.encoders[name].decode(features)
