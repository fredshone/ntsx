from pandas import Series
from typing import Optional, Iterable
from ntsx.encoders.utils import tokenize
from torch import Tensor


class BaseEncoder:

    def __init__(self, data: Iterable, name: Optional[str] = None):
        raise NotImplementedError(
            "BaseEncoder is an abstract class. Please use a concrete encoder."
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"

    def read_data(self, data: Iterable, name: Optional[str] = None) -> Series:

        if not isinstance(data, Series):
            print(f"Attempting to convert data ({type(data)}) to numpy array.")
            data = Series(data)

        self.name = name
        if name is None and isinstance(data, Series):
            self.name = data.name
        return data

    def encode(self, data: Iterable) -> Tensor:
        raise NotImplementedError("Encode method not implemented.")

    def decode(self, data: Iterable) -> Series:
        raise NotImplementedError("Decode method not implemented.")


class ContinuousEncoder(BaseEncoder):
    def __init__(
        self, data: Iterable, name: Optional[str] = None, verbose: bool = False
    ):
        """ContinuousEncoder is used to encode continuous data to a range between 0 and 1.

        Args:
            data (Iterable): input data to be encoded
            name (str, optional): name of the encoder. Defaults to None.
            verbose (bool, optional): print the encoder configuration. Defaults to False.
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
            f"{self}: min: {self.mini}, max: {self.maxi}, range: {self.range}, dtype: {self.dtype}"
        )

    def encode(self, data: Series) -> Tensor:
        data = Tensor(data.to_numpy())
        return (data - self.mini) / self.range

    def decode(self, data: Tensor) -> Series:
        data = Series(data)
        new = data * self.range + self.mini
        return new.astype(self.dtype)


class TimeEncoder(BaseEncoder):
    def __init__(
        self,
        data: Iterable,
        name: Optional[str] = None,
        min_value: float = 0,
        day_range: float = 1440,
        verbose: bool = False,
    ):
        """TimeEncoder is used to encode continuous data to a range between 0 and 1.

        Args:
            data (Iterable): input data to be encoded
            name (str, optional): name of the encoder. Defaults to None.
            min_value (float, optional): minimum value of the encoder. Defaults to 0.
            day_range (float, optional): range of the encoder. Defaults to 1440.
            verbose (bool, optional): print the encoder configuration. Defaults to False.
        Raises:
            UserWarning: If the data is not of type int or float.
        """
        data = self.read_data(data, name)
        if data.dtype.kind not in "fi":
            raise UserWarning("TimeEncoder only supports float and int data types.")
        self.mini = min_value
        self.range = day_range
        self.mean = data.mean()
        self.std = data.std()
        if self.range == 0:
            raise UserWarning("Data has no range. Cannot encode.")
        self.dtype = data.dtype

        self.encoding = "time"
        self.size = 1

        if verbose:
            print(
                f"{self.name}: min: {self.mini}, range: {self.range}, dtype: {self.dtype}"
            )

    def encode(self, data: Series) -> Tensor:
        data = Tensor(data.to_numpy())
        return (data - self.mini) / self.range

    def decode(self, data: Tensor) -> Series:
        data = Series(data)
        new = data * self.range + self.mini
        return new.astype(self.dtype)


class CategoricalTokeniser(BaseEncoder):
    def __init__(
        self, data: Iterable, name: Optional[str] = None, verbose: bool = False
    ):
        """CategoricalEncoder is used to encode categorical data as integers from 0 to N.
        Where N is the number of unique categories.
        Args:
            data (Iterable): input data to be encoded
            name (str, optional): name of the encoder. Defaults to None.
            verbose (bool, optional): print the encoder configuration. Defaults to False.
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

        if verbose:
            print(
                f"{self.name}: size: {self.size}, categories: {self.mapping}, dtype: {self.dtype}"
            )
            if self.size > 20:
                print(
                    f">>> Warning: CategoricalEncoder has more than 20 categories ({self.size})). <<<"
                )

    def encode(self, data: Series) -> Tensor:
        data = Series(data)
        return tokenize(data, self.mapping)[0]

    def decode(self, data: Tensor, safe: bool = True) -> Series:
        data = Series(data)
        reverse_mapping = {v: k for k, v in self.mapping.items()}
        if safe:
            missing = set(data.unique()) - set(reverse_mapping.keys())
            if missing:
                raise UserWarning(
                    f"Missing categories in data: {missing}. Please check your encoding."
                )
        return data.map(reverse_mapping).astype(self.dtype)
