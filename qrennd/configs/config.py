"""Parameter configuration (Config) class."""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Type, TypeVar

import yaml

T = TypeVar("T", bound="Config")


def range_constructor(loader, node):
    args = loader.construct_sequence(node)
    return list(range(*args))


yaml.add_constructor("!range", range_constructor)


class Config:
    """Config class containing data, train and model hyperparameters and checkpoint/summary directories."""

    def __init__(
        self,
        experiment: str,
        run: str,
        setup: Dict[str, dict],
        data_dir: str,
        output_dir: Optional[str] = None,
        add_timestamp: bool = True,
        *,
        seed: Optional[int] = None,
        init_weights: Optional[str] = None,
    ) -> None:
        self.experiment = experiment

        if add_timestamp:
            current_time = datetime.now()
            timestamp = current_time.strftime("%Y%m%d-%H%M%S")

            self.run = f"{timestamp}_{run}"
        else:
            self.run = run

        self.model = setup["model"]
        self.train = setup["train"]
        self.dataset = setup["dataset"]

        self.data_dir = Path(data_dir)
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = None

        self.init_weights = init_weights

        self.seed = seed

    @property
    def experiment_dir(self) -> Path:
        return self.data_dir / self.experiment

    @property
    def run_dir(self) -> Path:
        if self.output_dir is None:
            raise ValueError("output_dir not specified in config.")
        return self.output_dir / self.experiment / self.run

    @property
    def log_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoint"

    @classmethod
    def from_yaml(
        cls: Type[T],
        filepath: str,
        data_dir: str,
        output_dir: str,
    ) -> T:
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
            with open(filepath, "r") as file:
                setup = yaml.full_load(file)
        except FileNotFoundError as error:
            raise ValueError(
                f"Invalid Config setup file provided:  {filepath}"
            ) from error

        try:
            metadata = setup.pop("metadata")

            experiment = metadata["experiment"]
            run = metadata["run"]
            init_weights = metadata.get("init_weights", None)
            seed = metadata.get("seed")

        except KeyError:
            raise ValueError("Invalid config file format.")

        return cls(
            experiment,
            run,
            setup,
            data_dir,
            output_dir,
            init_weights=init_weights,
            seed=seed,
        )

    def to_yaml(self, filepath: str) -> None:
        metadata = dict(
            experiment=self.experiment,
            run=self.run,
            init_weights=self.init_weights,
            seed=self.seed,
        )
        setup = dict(
            metadata=metadata,
            train=self.train,
            dataset=self.dataset,
            model=self.model,
        )

        with open(filepath, mode="wt", encoding="utf-8") as file:
            yaml.dump(setup, file)
