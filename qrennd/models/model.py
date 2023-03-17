"""The 2-layer LSTM RNN model use for the decoder."""
from itertools import repeat, chain
from typing import Callable, Dict, Optional, Union, List

import numpy as np
from tensorflow import concat, keras

from ..configs import Config


def get_model(
    seq_size: List[int],
    vec_size: int,
    config: Config,
    optimizer: Optional[str] = None,
    loss: Optional[Dict[str, Union[str, Callable]]] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    metrics: Optional[Dict[str, str]] = None,
    name: Optional[str] = None,
) -> keras.Model:
    """
    get_model Returns the RNN decoder model.

    Parameters
    ----------
    seq_size : List[int]
        The shape of each sequence that is given to the LSTM or ConvLSTM layers.
    vec_size : Tuple[int]
        The size of the vector given directly to the evaluation layer.
    input_dtype : str
        Data type for the input of the model.
    config : Config
        The model configuartion, given as a Config dataclass. The configuation file
        should define the model property with following keys:
            lstm_units - the number of units each LSTM layer uses.
            eval_units - number of units in the hidden evaluation layer.
            output_units - the number of units in the output layer (for binary classification this should be 1).
        And a train dictiornary with the following keys:
            learning_rate - the learning rate of the Adam optimizer.
            dropout_rate - the dropout rate used after each LSTM layer and the hidden evaluation layer.
            l2_factor - the weight decay factor used to regularize the weights of the hidden evaluation layer.
    metrics : Optional[Dict[str, str]], optional
        Optional dictionary specifying the list of built-in metric names to be evaluatted by each model head during
        trainig and testing, by default None
    loss : Optional[Dict[str, Union[str, Callable]]], optional
        Optional dictionary of loss functions given as strings (name of the loss function) or
        functiton (Callable) for each head, by default None
    loss_weights : Optional[Dict[str, float]], optional
        Optional dictionary specifying the scalar coefficients (floats) to weight
        the loss contribution of each head output, by default None
    optimizer : Optional[str], optional
        The model optimizer given as a string (name of optimizer) or optimizer instance, by default None
    name : Optional[str], optional
        The name of the model, by default None

    Returns
    -------
    tensorflow.keras.Model
        The build and compiled model.
    """
    recurrent_input = keras.layers.Input(
        shape=(None, *seq_size),
        dtype="float32",
        name="lstm_input",
    )
    eval_input = keras.layers.Input(
        shape=(vec_size,),
        dtype="float32",
        name="eval_input",
    )

    if "ConvLSTM_units" in config.model:
        # Get ConvLSTM layers
        convlstm_units = config.model["ConvLSTM_units"]
        dropout_rates = config.model.get("ConvLSTM_dropout_rates")
        convlstm_kernels = config.model["ConvLSTM_kernels"]
        to_LSTM_input = "LSTM_units" in config.model
        convlstm_layers = conv_lstm_network(
            convlstm_units=convlstm_units,
            convlstm_kernels=convlstm_kernels,
            dropout_rates=dropout_rates,
            to_LSTM_input=to_LSTM_input,
        )
        # Apply ConvLSTM layers
        conv_output = next(convlstm_layers)(recurrent_input)
        for layer in convlstm_layers:
            conv_output = layer(conv_output)

        # Reshape for next layer
        if to_LSTM_input:
            # reshape from [shots, timesteps, filters, rows, cols]
            # into [shots, timesteps, dim]
            reshape_layer = keras.layers.Reshape(
                (conv_output.shape[1], np.product(conv_output.shape[2:]))
            )
            lstm_input = reshape_layer(conv_output)
        else:
            # reshape from [shots, filters, rows, cols] into [shots, dim]
            flatten_layer = keras.layers.Flatten()
            recurrent_output = flatten_layer(conv_output)
    else:
        lstm_input = recurrent_input

    if "LSTM_units" in config.model:
        # Get LSTM layers
        lstm_units = config.model["LSTM_units"]
        dropout_rates = config.model.get("LSTM_dropout_rates")
        lstm_layers = lstm_network(
            lstm_units=lstm_units,
            dropout_rates=dropout_rates,
        )
        # Apply ConvLSTM layers
        recurrent_output = next(lstm_layers)(lstm_input)
        for layer in lstm_layers:
            recurrent_output = layer(recurrent_output)

    # Get evaluation layers
    l2_factor = config.model.get("l2_factor")
    eval_units = config.model.get("eval_units", 64)
    dropout_rate = config.model.get("eval_dropout_rate")
    output_units = config.model.get("output_units", 1)

    main_eval_layers = evaluation_network(
        eval_units=eval_units,
        output_units=output_units,
        dropout_rate=dropout_rate,
        l2_factor=l2_factor,
        name="main",
    )
    aux_eval_layers = evaluation_network(
        eval_units=eval_units,
        output_units=output_units,
        dropout_rate=dropout_rate,
        l2_factor=l2_factor,
        name="aux",
    )
    # Apply evaluation layers
    main_input = concat((recurrent_output, eval_input), axis=1)
    main_output = next(main_eval_layers)(main_input)
    for layer in main_eval_layers:
        main_output = layer(main_output)

    aux_output = next(aux_eval_layers)(recurrent_output)
    for layer in aux_eval_layers:
        aux_output = layer(aux_output)

    # Compile model
    model = keras.Model(
        inputs=[recurrent_input, eval_input],
        outputs=[main_output, aux_output],
        name=name or "decoder_model",
    )

    if optimizer is None:
        opt_params = config.train.get("optimizer")
        if opt_params is not None:
            optimizer = keras.optimizers.Adam(**opt_params)
        else:
            optimizer = keras.optimizers.Adam()

    if loss is None:
        loss = config.train.get("loss")

    if loss_weights is None:
        loss_weights = config.train.get("loss_weights")

    if metrics is None:
        metrics = config.train.get("metrics")

    model.compile(
        optimizer=optimizer,
        loss=loss,
        loss_weights=loss_weights,
        metrics=metrics,
    )

    # Load initial weights for the model if necessary
    init_weights = config.init_weights
    if init_weights is not None:
        try:
            experiment_dir = config.output_dir / config.experiment
            model.load_weights(experiment_dir / init_weights)
        except FileNotFoundError as error:
            raise ValueError(
                "Invalid initial weights in configuration file."
            ) from error

    return model


def lstm_network(
    lstm_units: List[int],
    dropout_rates: List[float] = None,
):
    num_layers = len(lstm_units)
    if dropout_rates is None:
        dropout_rates = list(repeat(None, len(lstm_units)))

    if len(dropout_rates) != num_layers:
        raise ValueError(
            f"Mismatch between the number of LSTM layers ({num_layers})"
            "and the number of LSTM dropout rate after each layer."
        )
    inds = range(1, num_layers + 1)

    for ind, units, rate in zip(inds, lstm_units, dropout_rates):
        return_sequences = ind != num_layers
        lstm_layer = keras.layers.LSTM(
            units=units,
            return_sequences=return_sequences,
            name=f"LSTM_{ind}",
        )
        yield lstm_layer

        if rate is not None:
            dropout_layer = keras.layers.Dropout(
                rate=rate,
                name=f"dropout_LSTM_{ind}",
            )
            yield dropout_layer

    act_layer = keras.layers.Activation(
        activation="relu",
        name="relu_lstm",
    )
    yield act_layer


def conv_lstm_network(
    convlstm_units: List[int],
    convlstm_kernels: List[int],
    dropout_rates: Optional[List[float]] = None,
    to_LSTM_input: Optional[Union[bool, List[int]]] = False,
):
    """
    to_LSTM_input : bool or List[int]
        If False, adds Flatten layer so the output can be passed
        to the evaluation layers.
        If List[int], this list corresponds to [timesteps, ]
    """
    num_layers = len(convlstm_units)
    if dropout_rates is None:
        dropout_rates = repeat(None, len(convlstm_units))

    if len(dropout_rates) != num_layers:
        raise ValueError(
            f"Mismatch between the number of ConvLSTM layers ({num_layers})"
            "and the number of ConvLSTM dropout rate after each layer."
        )
    if len(convlstm_kernels) != num_layers:
        raise ValueError(
            f"Mismatch between the number of ConvLSTM layers ({num_layers})"
            "and the number of ConvLSTM kernel sizes."
        )
    inds = range(1, num_layers + 1)

    for ind, units, rate, kernel_size in zip(
        inds, convlstm_units, dropout_rates, convlstm_kernels
    ):
        return_sequences = (ind != num_layers) or to_LSTM_input
        convlstm_layer = keras.layers.ConvLSTM2D(
            filters=units,
            kernel_size=kernel_size,
            data_format="channels_first",
            return_sequences=return_sequences,
            name=f"ConvLSTM_{ind}",
        )
        yield convlstm_layer

        if rate is not None:
            dropout_layer = keras.layers.Dropout(
                rate=rate,
                name=f"dropout_ConvLSTM_{ind}",
            )
            yield dropout_layer

    act_layer = keras.layers.Activation(
        activation="relu",
        name="relu_convlstm",
    )
    yield act_layer


def evaluation_network(
    eval_units: int,
    output_units: int,
    dropout_rate: float = None,
    l2_factor: float = None,
    name: str = "eval",
):
    """
    Adds two evaluation layers
    """
    if l2_factor is not None:
        regularizer = keras.regularizers.L2(l2_factor)
    else:
        regularizer = None

    # First evaluation layer
    dense_layer = keras.layers.Dense(
        units=eval_units,
        activation="relu",
        kernel_regularizer=regularizer,
        name=f"{name}_dense",
    )
    yield dense_layer

    if dropout_rate is not None:
        dropout_layer = keras.layers.Dropout(
            rate=dropout_rate,
            name=f"dropout_{name}_dense",
        )
        yield dropout_layer

    # Second evaluation layer
    output_layer = keras.layers.Dense(
        units=output_units,
        activation="sigmoid",
        name=f"{name}_output",
    )
    yield output_layer
