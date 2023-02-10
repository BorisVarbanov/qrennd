from typing import Any, Dict, Tuple

from tensorflow.keras.callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
)

from qrennd.configs import Config


def get_checkpoint_filename(params: Dict[str, Any]) -> str:
    if params["save_best_only"]:
        filename = "weights.hdf5"
        return filename

    filename = f"weights-{{epoch}}-{{{params['monitor']}}}.hdf5"
    return filename


def get_callbacks(config: Config) -> Tuple[Callback]:
    params = config.train.get("callbacks")

    checkpoint_filename = get_checkpoint_filename(params["checkpoint"])
    model_checkpoint = ModelCheckpoint(
        filepath=config.checkpoint_dir / checkpoint_filename,
        **params["checkpoint"],
    )

    early_stop = EarlyStopping(
        **params["early_stop"],
    )

    csv_logs = CSVLogger(
        filename=config.log_dir / "training.log",
        **params["csv_log"],
    )

    return model_checkpoint, early_stop, csv_logs
