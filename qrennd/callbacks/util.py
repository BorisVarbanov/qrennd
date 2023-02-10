from qrennd.configs import Config

from tensorflow import keras


def get_callbacks(config: Config):
    callbacks = []
    params = config.train.get("callbacks")

    checkpoint_params = params["checkpoint"]

    if checkpoint_params["save_best_only"]:
        checkpoint = "weights.hdf5"
    else:
        checkpoint = "weights-{epoch:02d}-{val_loss:.5f}.hdf5"

    output_dir = config.output_dir

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=config.output_dir / checkpoint_str,
        params,
    )
    early_stop = keras.callbacks.EarlyStopping(
        **params["early_stop"]
    )

    logs_filename = config.log_dir / "training.log"
    csv_logs = keras.callbacks.CSVLogger(
        filename=logs_filename,
        append=False,
    )

    return callbacks
