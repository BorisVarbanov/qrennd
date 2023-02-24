from typing import Dict

from ..configs import Config
from ..layouts import Layout
from .generators import DataGeneratorGoogle, dataset_genereator
from .preprocessing import preprocess_data
from .sequences import RaggedSequence


def load_datasets(config: Config, layout: Layout, dataset_name: str):
    batch_size = config.train["batch_size"]
    experiment_name = config.dataset["folder_format_name"]

    rot_basis = config.dataset["rot_basis"]
    basis = "X" if rot_basis else "Z"
    stab_type = "x_type" if rot_basis else "z_type"

    dataset_dir = config.experiment_dir / dataset_name
    dataset_params = config.dataset[dataset_name]

    dataset_gen = dataset_genereator(
        dataset_dir, experiment_name, basis, **dataset_params
    )
    proj_matrix = layout.projection_matrix(stab_type)

    lstm_input = config.dataset["lstm_input"]
    eval_input = config.dataset["lstm_input"]

    generator = (
        preprocess_data(dataset, lstm_input, eval_input, proj_matrix)
        for dataset in dataset_gen
    )
    return RaggedSequence.from_generator(generator, batch_size)


def load_datasets_google(
    config: Config, layout: Layout
) -> Dict[str, DataGeneratorGoogle]:
    batch_size = config.train["batch_size"]

    lstm_input = config.dataset["lstm_input"]
    eval_input = config.dataset["eval_input"]
    folder_format_name = config.dataset["folder_format_name"]

    rot_basis = config.dataset["rot_basis"]

    if eval_input == "defects":
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
            generator = DataGeneratorGoogle(
                dirpath=dataset_dir,
                batch_size=batch_size,
                lstm_input=lstm_input,
                eval_input=eval_input,
                proj_matrix=proj_matrix,
                folder_format_name=folder_format_name,
                rot_basis=rot_basis,
                **dataset_params,
            )

            generators.append(generator)
        except FileNotFoundError as error:
            raise ValueError("Invalid experiment data directory") from error

    return generators
