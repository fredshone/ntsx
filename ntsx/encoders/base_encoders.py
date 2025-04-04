from pandas import Series
from typing import Optional, Iterable
from ntsx.encoders.utils import tokenize


class BaseEncoder:
    def read_data(self, data: Iterable, name: Optional[str] = None) -> Series:

        if not isinstance(data, Series):
            print(f"Attempting to convert data ({type(data)}) to numpy array.")
            data = Series(data)

        self.name = name
        if name is None and isinstance(data, Series):
            self.name = data.name
        return data


class ContinuousEncoder(BaseEncoder):
    def __init__(self, data: Iterable, name: Optional[str] = None):
        """ContinuousEncoder is used to encode continuous data to a range between 0 and 1.

        Args:
            data (Iterable): input data to be encoded
            name (str, optional): name of the encoder. Defaults to None.
        Raises:
            UserWarning: If the data is not of type int or float.
        """
        data = self.read_data(data, name)

        if data.dtype.kind not in "fi":
            raise UserWarning(
                "ContinuousEncoder only supports float and int data types."
            )
        self.mini = data.min()
        self.maxi = data.max()
        self.range = self.maxi - self.mini
        self.mean = data.mean()
        self.std = data.std()
        if self.range == 0:
            raise UserWarning("Data has no range. Cannot encode.")
        self.dtype = data.dtype

        self.encoding = "continuous"
        self.size = 1

        print(
            f"""ContinuousEncoder({self.name}):
            min: {self.mini}, max: {self.maxi},
            range: {self.range}
            mean: {self.mean}, std: {self.std}
            dtype: {self.dtype}
"""
        )

    def encode(self, data: Iterable) -> Series:
        data = Series(data)
        return (data - self.mini) / self.range

    def decode(self, data: Iterable) -> Series:
        data = Series(data)
        new = data * self.range + self.mini
        return new.astype(self.dtype)


class TimeEncoder(BaseEncoder):
    def __init__(
        self,
        data: Iterable,
        name: Optional[str] = None,
        min_value: float = 0,
        day_value: float = 1440,
    ):
        """TimeEncoder is used to encode continuous data to a range between 0 and 1.

        Args:
            data (Iterable): input data to be encoded
            name (str, optional): name of the encoder. Defaults to None.
            min_value (float, optional): minimum value of the encoder. Defaults to 0.
            day_value (float, optional): range of the encoder. Defaults to 1440.
        Raises:
            UserWarning: If the data is not of type int or float.
        """
        data = self.read_data(data, name)
        if data.dtype.kind not in "fi":
            raise UserWarning("TimeEncoder only supports float and int data types.")
        self.mini = min_value
        self.range = day_value
        self.mean = data.mean()
        self.std = data.std()
        if self.range == 0:
            raise UserWarning("Data has no range. Cannot encode.")
        self.dtype = data.dtype

        self.encoding = "time"
        self.size = 1

        print(
            f"""TimeEncoder({self.name}):
            min: {self.mini}, range: {self.range}
            mean: {self.mean}, std: {self.std}
            dtype: {self.dtype}
"""
        )

    def encode(self, data: Iterable) -> Series:
        data = Series(data)
        return (data - self.mini) / self.range

    def decode(self, data: Iterable) -> Series:
        data = Series(data)
        new = data * self.range + self.mini
        return new.astype(self.dtype)


class CategoricalTokeniser(BaseEncoder):
    def __init__(self, data: Iterable, name: Optional[str] = None):
        """CategoricalEncoder is used to encode categorical data as integers from 0 to N.
        Where N is the number of unique categories.
        Args:
            data (Iterable): input data to be encoded
            name (str, optional): name of the encoder. Defaults to None.
        Raises:
            UserWarning: If the data is not of type int or object.
        """
        data = self.read_data(data, name)
        if data.dtype.kind not in "iO":
            raise UserWarning(
                "CategoricalEncoder only supports object and int data types."
            )
        self.dtype = data.dtype
        _, self.mapping = tokenize(data)

        self.encoding = "categorical"
        self.size = len(self.mapping)

        print(
            f"""CategoricalEncoder({self.name}):
            size: {self.size}
            categories: {self.mapping}
            """
        )
        if self.size > 20:
            print(
                f">>> Warning: CategoricalEncoder has more than 20 categories ({self.size})). <<<"
            )

    def encode(self, data: Iterable) -> Series:
        data = Series(data)
        return Series(tokenize(data, self.mapping)[0]).astype("int")

    def decode(self, data: Iterable) -> Series:
        data = Series(data)
        return data.map({v: k for k, v in self.mapping.items()}).astype(self.dtype)
