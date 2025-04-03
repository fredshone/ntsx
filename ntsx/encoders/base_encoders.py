from pandas import Series
from typing import Optional
from ntsx.encoders.utils import tokenize


class ContinuousEncoder:
    def __init__(self, data: Series, name: Optional[str] = None):
        """ContinuousEncoder is used to encode continuous data to a range between 0 and 1.

        Args:
            data (Series): input data to be encoded
            name (str, optional): name of the encoder. Defaults to None.
        Raises:
            UserWarning: If the data is not of type int or float.
        """
        if data.dtype.kind not in "fi":
            raise UserWarning(
                "ContinuousEncoder only supports float and int data types."
            )
        self.name = name
        if name is None:
            self.name = data.name
        self.mini = data.min()
        self.maxi = data.max()
        self.range = self.maxi - self.mini
        self.mean = data.mean()
        self.std = data.std()
        if self.range == 0:
            raise UserWarning("Data has no range. Cannot encode.")
        self.dtype = data.dtype
        print(
            f"""ContinuousEncoder({self.name}):
            min: {self.mini}, max: {self.maxi},
            range: {self.range}
            mean: {self.mean}, std: {self.std}
            dtype: {self.dtype}
"""
        )

    def encode(self, data: Series) -> Series:
        return (data - self.mini) / self.range

    def decode(self, data: Series) -> Series:
        new = data * self.range + self.mini
        return new.astype(self.dtype)


class TimeEncoder:
    def __init__(
        self,
        data: Series,
        name: Optional[str] = None,
        min_value: float = 0,
        day_value: float = 1440,
    ):
        """TimeEncoder is used to encode continuous data to a range between 0 and 1.

        Args:
            data (Series): input data to be encoded
            name (str, optional): name of the encoder. Defaults to None.
            min_value (float, optional): minimum value of the encoder. Defaults to 0.
            day_value (float, optional): range of the encoder. Defaults to 1440.
        Raises:
            UserWarning: If the data is not of type int or float.
        """
        if data.dtype.kind not in "fi":
            raise UserWarning("TimeEncoder only supports float and int data types.")
        self.name = name
        if name is None:
            self.name = data.name
        self.mini = min_value
        self.range = day_value
        self.mean = data.mean()
        self.std = data.std()
        if self.range == 0:
            raise UserWarning("Data has no range. Cannot encode.")
        self.dtype = data.dtype
        print(
            f"""TimeEncoder({self.name}):
            min: {self.mini}, range: {self.range}
            mean: {self.mean}, std: {self.std}
            dtype: {self.dtype}
"""
        )

    def encode(self, data: Series) -> Series:
        return (data - self.mini) / self.range

    def decode(self, data: Series) -> Series:
        new = data * self.range + self.mini
        return new.astype(self.dtype)


class CategoricalTokeniser:
    def __init__(self, data: Series, name: Optional[str] = None):
        """CategoricalEncoder is used to encode categorical data as integers from 0 to N.
        Where N is the number of unique categories.
        Args:
            data (Series): input data to be encoded
            name (str, optional): name of the encoder. Defaults to None.
        Raises:
            UserWarning: If the data is not of type int or object.
        """
        if data.dtype.kind not in "iO":
            raise UserWarning(
                "CategoricalEncoder only supports object and int data types."
            )
        self.name = name
        if name is None:
            self.name = data.name
        self.dtype = data.dtype
        _, self.mapping = tokenize(data)
        print(
            f"""CategoricalEncoder({self.name}):
            categories: {self.mapping}
            """
        )

    def encode(self, data: Series) -> Series:
        return Series(tokenize(data, self.mapping)[0]).astype("int")

    def decode(self, data: Series) -> Series:
        return data.map({v: k for k, v in self.mapping.items()}).astype(self.dtype)
