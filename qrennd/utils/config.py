"""Parameter configuration (Config) class."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, TypeVar

import yaml

T = TypeVar("T", bound="Config")


@dataclass
class Config:
    """Config class containing data, train and model hyperparameters and checkpoint/summary directories."""

    exp_name: str
    data: dict
    train: dict
    model: dict
    summary_dir: Path = field(repr=False, init=False)
    checkpoint_dir: Path = field(repr=False, init=False)

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
            setup = yaml.safe_load(file)

        exp_name = setup.get("exp_name")
        data = setup.get("data")
        train = setup.get("train")
        model = setup.get("model")

        return cls(exp_name, data, train, model)
