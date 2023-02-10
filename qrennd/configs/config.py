"""Parameter configuration (Config) class."""
from os import error
from dataclasses import dataclass, field
from typing import Type, TypeVar, Dict

import yaml

T = TypeVar("T", bound="Config")


def range_constructor(loader, node):
    args = loader.construct_sequence(node)
    return list(range(*args))


yaml.add_constructor("!range", range_constructor)


@dataclass
class Config:
    """Config class containing data, train and model hyperparameters and checkpoint/summary directories."""

    metadata: Dict[str, str]

    dataset: dict
    train: dict
    model: dict

    data_dir: str = field(default=None)
    output_dir: str = field(default=None)

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
        try:
            with open(filename, "r") as file:
                setup = yaml.full_load(file)
        except error:
            raise ValueError(f"Invalid Config setup file provided:  {filename}")

        attributes = ("metadata", "dataset", "train", "model")
        args = {}

        for attr in attributes:
            try:
                args[attr] = setup[attr]
            except KeyError:
                raise ValueError("Invalid configuration file format.")

        return cls(**args)

    def to_yaml(self, filepath: str) -> None:
        attributes = ("metadata", "dataset", "train", "model")
        data = {attr: getattr(self, attr) for attr in attributes}

        with open(filepath, mode="wt", encoding="utf-8") as file:
            yaml.dump(data, file)
