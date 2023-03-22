from ..configs import Config
from ..layouts import Layout
from .generators import dataset_generator
from .preprocessing import to_defects, to_measurements, to_model_input, to_syndromes
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
        processed_gen = (to_measurements(dataset) for dataset in dataset_gen)
    elif input_type == "syndromes":
        processed_gen = (to_syndromes(dataset, proj_matrix) for dataset in dataset_gen)
    elif input_type == "defects":
        processed_gen = (to_defects(dataset, proj_matrix) for dataset in dataset_gen)
    else:
        raise ValueError(
            f"Unknown input data type {input_type}, the possible "
            "options are 'measurements', 'syndromes', 'defects' and 'MWPM'."
        )

    # Process for keras.model input
    exp_matrix = layout.expansion_matrix() if config.model["ConvLSTM"] else None
    input_gen = (
        to_model_input(lstm_inputs, eval_inputs, log_errors, exp_matrix)
        for lstm_inputs, eval_inputs, log_errors in processed_gen
    )

    return RaggedSequence.from_generator(input_gen, batch_size)
