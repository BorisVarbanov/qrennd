from typing import Dict

from ..configs import Config
from ..layouts import Layout
from .generator import DataGenerator


def load_datasets(config: Config, layout: Layout) -> Dict[str, DataGenerator]:
    batch_size = config.train["batch_size"]

    lstm_input = config.dataset["lstm_input"]
    eval_input = config.dataset["eval_input"]

    if eval_input == "defects":
        rot_basis = config.dataset["rot_basis"]
        proj_matrix = layout.projection_matrix(
            stab_type="x_type" if rot_basis else "z_type"
        )
    else:
        proj_matrix = None

    generators = []

    for dataset_name in ("train", "dev"):
        dataset_dir = config.experiment_dir / dataset_name

        dataset_params = config.dataset[dataset_name]

        try:
            generator = DataGenerator(
                dirpath=dataset_dir,
                batch_size=batch_size,
                lstm_input=lstm_input,
                eval_input=eval_input,
                proj_matrix=proj_matrix,
                **dataset_params,
            )

            generators.append(generator)
        except FileNotFoundError as error:
            raise ValueError("Invalid experiment data directory") from error

    return generators
