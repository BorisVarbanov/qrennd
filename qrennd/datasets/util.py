from ..configs import Config
from ..layouts import Layout
from .generators import dataset_genereator
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
