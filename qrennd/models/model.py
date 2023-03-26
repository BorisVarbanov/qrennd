"""The 2-layer LSTM RNN model use for the decoder."""
from itertools import repeat
from typing import Callable, Dict, List, Optional, Union

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
        name="recurrent_input",
    )
    eval_input = keras.layers.Input(
        shape=(vec_size,),
        dtype="float32",
        name="eval_input",
    )

    inputs = [recurrent_input, eval_input]
    outputs = []

    if conv_config := config.model["ConvLSTM"]:
        # Get ConvLSTM layers
        filters = conv_config["filters"]
        dropout_rates = conv_config.get("dropout_rates")
        kernel_sizes = conv_config["kernel_sizes"]
        convlstm_layers = conv_lstm_network(
            layer_filters=filters,
            kernel_sizes=kernel_sizes,
            dropout_rates=dropout_rates,
            return_sequences=config.model["LSTM"],
        )
        # Apply ConvLSTM layers
        conv_output = next(convlstm_layers)(recurrent_input)
        for layer in convlstm_layers:
            conv_output = layer(conv_output)

        # Reshape for next layer
        if config.model["LSTM"]:
            reshape_layer = keras.layers.Reshape(
                (-1, np.product(conv_output.shape[2:]))
            )
            lstm_input = reshape_layer(conv_output)
        else:
            flatten_layer = keras.layers.Flatten()
            recurrent_output = flatten_layer(conv_output)
    else:
        # No ConvLSTM layers
        lstm_input = recurrent_input

    if config_lstm := config.model["LSTM"]:
        # Get LSTM layers
        units = config_lstm["units"]
        dropout_rates = config_lstm.get("dropout_rates")
        lstm_layers = lstm_network(
            layer_units=units,
            dropout_rates=dropout_rates,
        )
        # Apply ConvLSTM layers
        recurrent_output = next(lstm_layers)(lstm_input)
        for layer in lstm_layers:
            recurrent_output = layer(recurrent_output)

    if config.model["decode"]:
        prev_ouputs = recurrent_output[:, :-1]
        last_output = recurrent_output[:, -1]

        units = np.prod(seq_size)
        dense_layer = keras.layers.Dense(
            units=units,
            activation="sigmoid",
            name="rec_decoder",
        )

        decoder_layer = keras.layers.TimeDistributed(dense_layer)

        recurrent_predictions = decoder_layer(prev_ouputs)
        outputs.append(recurrent_predictions)

        eval_decoder_layer = keras.layers.Dense(
            units=vec_size,
            activation="sigmoid",
            name="eval_decoder",
        )
        eval_prediction = eval_decoder_layer(last_output)
        outputs.append(eval_prediction)
    else:
        last_output = recurrent_output

    # Get evaluation layers
    config_eval = config.model["eval"]
    units = config_eval["units"]
    l2_factor = config_eval.get("l2_factor")
    dropout_rates = config_eval.get("dropout_rate")

    main_eval_layers = evaluation_network(
        layer_units=units,
        dropout_rates=dropout_rates,
        l2_factor=l2_factor,
        name="main",
    )
    aux_eval_layers = evaluation_network(
        layer_units=units,
        dropout_rates=dropout_rates,
        l2_factor=l2_factor,
        name="aux",
    )
    # Apply evaluation layers
    main_input = concat((last_output, eval_input), axis=1)
    main_output = next(main_eval_layers)(main_input)
    for layer in main_eval_layers:
        main_output = layer(main_output)
    outputs.append(main_output)

    aux_output = next(aux_eval_layers)(last_output)
    for layer in aux_eval_layers:
        aux_output = layer(aux_output)
    outputs.append(aux_output)

    # Compile model
    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
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
    layer_units: List[int],
    dropout_rates: Optional[List[float]] = None,
):
    num_layers = len(layer_units)
    if dropout_rates is None:
        dropout_rates = list(repeat(None, len(layer_units)))

    if len(dropout_rates) != num_layers:
        raise ValueError(
            f"Mismatch between the number of LSTM layers ({num_layers})"
            "and the number of LSTM dropout rate after each layer."
        )
    inds = range(1, num_layers + 1)

    for ind, units, rate in zip(inds, layer_units, dropout_rates):
        lstm_layer = keras.layers.LSTM(
            units=units,
            return_sequences=ind != num_layers,
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
    layer_filters: List[int],
    kernel_sizes: List[int],
    dropout_rates: Optional[List[float]] = None,
    return_sequences: Optional[bool] = False,
):
    num_layers = len(layer_filters)
    if dropout_rates is None:
        dropout_rates = list(repeat(None, num_layers))

    if len(dropout_rates) != num_layers:
        raise ValueError(
            f"Mismatch between the number of ConvLSTM layers ({num_layers})"
            "and the number of ConvLSTM dropout rate after each layer."
        )
    if len(kernel_sizes) != num_layers:
        raise ValueError(
            f"Mismatch between the number of ConvLSTM layers ({num_layers})"
            "and the number of ConvLSTM kernel sizes."
        )

    inds = range(1, num_layers + 1)
    for ind, filters, rate, kernel_size in zip(
        inds, layer_filters, dropout_rates, kernel_sizes
    ):
        convlstm_layer = keras.layers.ConvLSTM2D(
            filters=filters,
            kernel_size=kernel_size,
            data_format="channels_first",
            return_sequences=(ind != num_layers) or return_sequences,
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
    layer_units: List[int],
    dropout_rates: Optional[List[float]] = None,
    l2_factor: Optional[float] = None,
    name: Optional[str] = "eval",
):
    num_layers = len(layer_units)
    if dropout_rates is None:
        dropout_rates = list(repeat(None, num_layers))

    if len(dropout_rates) != num_layers:
        raise ValueError(
            f"Mismatch between the number of ConvLSTM layers ({num_layers})"
            "and the number of ConvLSTM dropout rate after each layer."
        )

    if l2_factor is not None:
        regularizer = keras.regularizers.L2(l2_factor)
    else:
        regularizer = None

    inds = range(1, num_layers + 1)
    for ind, units, rate in zip(inds, layer_units, dropout_rates):
        dense_layer = keras.layers.Dense(
            units=units,
            activation="relu",
            kernel_regularizer=regularizer,
            name=f"{name}_dense_{ind}",
        )
        yield dense_layer

        if rate is not None:
            dropout_layer = keras.layers.Dropout(
                rate=rate,
                name=f"dropout_{name}_dense_{ind}",
            )
            yield dropout_layer

    # Final evaluation layer
    output_layer = keras.layers.Dense(
        units=1,
        activation="sigmoid",
        name=f"{name}_output",
    )
    yield output_layer
