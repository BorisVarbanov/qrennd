"""Parameter configuration (Config) class."""
from dataclasses import dataclass, field, asdict
from typing import Type, TypeVar, Dict

import yaml

T = TypeVar("T", bound="Config")


def range_constructor(loader, node):
    args = loader.construct_sequence(node)
    return range(*args)


yaml.add_constructor("!range", range_constructor)


@dataclass
class Config:
    """Config class containing data, train and model hyperparameters and checkpoint/summary directories."""

    metadata: Dict[str, str]

    dataset: dict
    train: dict
    model: dict

    data_dir: str = field(repr=False, default=None)
    output_dir: str = field(repr=False, default=None)

    @classmethod
    def from_yaml(cls: Type[T], filename: str) -> T:
        """
        from_yaml Create new qrennd.utils.Config instance from YAML configuarion file.

        Parameters
        ----------
        filename : str
            The YAML file name.

        Returns
        -------
        T
            The initialised qrennd.utils.Config object based on the yaml.
        """
        with open(filename, "r") as file:
            setup = yaml.full_load(file)

        arg_names = ("metadata", "dataset", "train", "model")
        args = {}

        for name in arg_names:
            try:
                val = setup[name]
                args[name] = val
            except KeyError:
                raise ValueError("Invalid configuration file format.")

        return cls(**args)

    def to_yaml(self, filepath: str) -> None:
        data = asdict(self)

        with open(filepath, mode="wb") as file:
            yaml.dump(data, file, encoding="utf-8")
