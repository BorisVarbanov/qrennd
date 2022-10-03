# +
import pathlib
from math import floor
from datetime import datetime
from typing import Iterable, Optional, Tuple

import numpy as np
import xarray as xr
from xarray import DataArray

from tensorflow.keras import (
    Model,
    layers, 
    optimizers, 
    callbacks, 
    losses, 
    metrics
)

from sklearn.model_selection import train_test_split


# -

def percent(number):
    return number*1e2


def get_model(
    layer_units : Iterable[int], 
    num_states: Optional[int] = 3,
    dropout_rate: Optional[float] = None,
    optimizer: Optional[str] = None,
    loss: Optional[str] = None,
    metrics: Optional[str] = None,
) -> Model:
    input_shape = (2, )
    inputs = layers.Input(shape=input_shape, dtype="float32", name="IQ signal")

    outputs = None
    for layer_ind, units in enumerate(layer_units):
        layer_name = f"dense_{layer_ind + 1}"
        
        dense_layer = layers.Dense(
            units=units,
            activation="relu",
            name=layer_name,
        )
        outputs = dense_layer(outputs if outputs is not None else inputs)
        
        if dropout_rate:
            dropout_layer = layers.Dropout(
                rate=dropout_rate, 
                name=f"{layer_name}_dropout(r={dropout_rate})"
            )
            outputs = dropout_layer(outputs)

    prediction_layer = layers.Dense(
        units=num_states,
        activation="softmax",
        name="prediction_layer",
    )
    outputs = prediction_layer(outputs)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="readout_classifier",
    )

    model.compile(
        optimizer = optimizer or "adam",
        loss = loss or "sparse_categorical_crossentropy",
        metrics = metrics or "sparse_categorical_accuracy",
    )
    return model


def split_data(dataset: DataArray, dev_fraction: float, test_fraction: float, seed: Optional[int] = None) -> Tuple[DataArray]:
    arr_size = dataset.shape[0]
    
    dev_size = floor(arr_size * dev_fraction)
    test_size = floor(arr_size * test_fraction)
    train_size = arr_size - dev_size - test_size

    rng = np.random.default_rng(seed=seed)
    inds = rng.permutation(arr_size)
    
    train_inds, dev_inds, test_inds = np.split(inds, [train_size, train_size+dev_size])
    return dataset[train_inds], dataset[dev_inds], dataset[test_inds]


# %load_ext tensorboard

# +
NOTEBOOK_DIR = pathlib.Path.cwd()

DATA_DIR = NOTEBOOK_DIR / "data"
if not DATA_DIR.exists():
    raise ValueError("Data directory does not exist.")
    
cur_datetime = datetime.now()
datetime_str = cur_datetime.strftime('%Y%m%d-%H%M%S')
LOG_DIR = NOTEBOOK_DIR / f".logs/{datetime_str}"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# +
QUBIT = "X3"

file_name = f"{QUBIT}_classifier_calibration_dataset.nc"
exp_data = xr.load_dataarray(DATA_DIR / file_name)

# +
seed = 42
dev_fraction = 0.2
test_fraction = 0.2

dataset = exp_data.stack(exp_shot = ("state", "shot"))
dataset = dataset.transpose("exp_shot", "comp")

test_dataset, dev_dataset, train_dataset = split_data(dataset, dev_fraction, test_fraction, seed)
# -

# # Build the Feedforward Neural Network

# +
#LAYERS_UNITS = [32, 16, 8]
LAYERS_UNITS = [16, 8]
LEARNING_RATE = 0.0005

BATCH_SIZE = 64
# -

optimizer = optimizers.Adam(LEARNING_RATE)
model = get_model(LAYERS_UNITS, optimizer=optimizer, metrics="sparse_categorical_accuracy", loss = "sparse_categorical_crossentropy")

model.summary()

# # Train and validate the model

# %tensorboard --logdir={LOG_DIR}

# +
tensorboard = callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

model.fit(
    x = train_dataset.data,
    y = train_dataset.state.data, 
    validation_data = (
        dev_dataset.data, 
        dev_dataset.state.data
    ),
    batch_size = 64,
    epochs = 200,
    callbacks = [
        tensorboard,
    ]
)
# -

dnn_loss, dnn_accuracy = model.evaluate(
    x = test_dataset.data,
    y = test_dataset.state.data,
    batch_size = BATCH_SIZE
)

print(f"Total FNN classifier accuracy on the test dataset: {percent(dnn_accuracy):.3f} %")

# # Comparison with the standard LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda_classifier = LinearDiscriminantAnalysis(
    solver="svd", 
    shrinkage=None, 
    tol=1e-4
)

# +
lda_classifier.fit(train_dataset, train_dataset.state)

lda_accuracy = lda_classifier.score(test_dataset, test_dataset.state)
print(f"Total LDA classifier accuracy on the test dataset: {percent(lda_accuracy):.3f} %")
# -


