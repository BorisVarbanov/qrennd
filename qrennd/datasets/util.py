from ..configs import Config
from ..layouts import Layout
from .generators import dataset_generator
from .preprocessing import (
    to_model_input,
    to_defect_probs_leakage_IQ,
)
from .sequences import RaggedSequence, Sequence


def load_datasets(
    config: Config,
    layout: Layout,
    dataset_name: str,
    concat: bool = True,
):
    batch_size = config.train["batch_size"]
    model_type = config.model["type"]
    predict_defects = model_type == "LSTM_decoder"
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
    if input_type == "IQ":
        digitization = config.dataset.get("digitization")
        leakage = config.dataset.get("leakage")
        processed_gen = (
            to_defect_probs_leakage_IQ(
                dataset, proj_matrix, digitization=digitization, leakage=leakage
            )
            for dataset in dataset_gen
        )
    else:
        raise ValueError(
            f"Unknown input data type {input_type}, the possible "
            "options are 'measurements', 'syndromes', 'defects' and 'prob_defects'."
        )

    # Process for keras.model input
    conv_models = ("ConvLSTM", "Conv_LSTM")
    float_inputs = ("IQ",)
    exp_matrix = layout.expansion_matrix() if (model_type in conv_models) else None
    data_type = float if input_type in float_inputs else bool
    input_gen = (to_model_input(*arrs, exp_matrix, data_type) for arrs in processed_gen)

    if concat:
        return RaggedSequence.from_generator(input_gen, batch_size, predict_defects)

    sequences = (
        Sequence(*tensors, batch_size, predict_defects) for tensors in input_gen
    )
    return sequences
