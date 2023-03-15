import xarray as xr

from ..configs import Config
from ..layouts import Layout
from .generators import dataset_generator
from .preprocessing import (
    to_defects,
    to_measurements,
    to_syndromes,
    to_conv_input,
    to_model_input,
)
from .sequences import RaggedSequence


def load_datasets(config: Config, layout: Layout, dataset_name: str):
    batch_size = config.train["batch_size"]
    experiment_name = config.dataset["folder_format_name"]

    rot_basis = config.dataset["rot_basis"]
    basis = "X" if rot_basis else "Z"
    stab_type = "x_type" if rot_basis else "z_type"

    dataset_dir = config.experiment_dir / dataset_name
    dataset_params = config.dataset[dataset_name]

    dataset_gen = dataset_generator(
        dataset_dir, experiment_name, basis, **dataset_params
    )
    proj_matrix = layout.projection_matrix(stab_type)

    # Convert to desired input
    input_type = config.dataset["input"]
    if input_type == "measurements":
        dataset_gen = (to_measurements(dataset) for dataset in dataset_gen)
    elif input_type == "syndromes":
        dataset_gen = (to_syndromes(dataset, proj_matrix) for dataset in dataset_gen)
    elif input_type == "defects":
        dataset_gen = (to_defects(dataset, proj_matrix) for dataset in dataset_gen)
    else:
        raise ValueError(
            f"Unknown input data type {input_type}, the possible "
            "options are 'measurements', 'syndromes', 'defects' and 'MWPM'."
        )

    # Reshape if necessary
    if config.model["use_conv"]:
        expansion_matrix = layout.expansion_matrix()
        dataset_gen = (
            to_conv_input(lstm_inputs, eval_inputs, log_errors, expansion_matrix)
            for lstm_inputs, eval_inputs, log_errors in dataset_gen
        )

    # Process for keras.model input
    generator = (
        to_model_input(lstm_inputs, eval_inputs, log_errors)
        for lstm_inputs, eval_inputs, log_errors in dataset_gen
    )

    return RaggedSequence.from_generator(generator, batch_size)
