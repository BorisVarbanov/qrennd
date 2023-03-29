"""The 2-layer LSTM RNN model use for the decoder."""
from itertools import repeat
from typing import List, Optional

import numpy as np
from tensorflow import keras

from ..configs import Config


def get_model(
    seq_size: List[int],
    vec_size: int,
    config: Config,
    name: str = "decoder",
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
    name : str, optional
        The name of the model, by default 'decoder'

    Returns
    -------
    tensorflow.keras.Model
        The build and compiled model.
    """
    rec_input = keras.layers.Input(
        shape=(None, *seq_size),
        dtype="float32",
        name="rec_input",
    )
    eval_input = keras.layers.Input(
        shape=(vec_size,),
        dtype="float32",
        name="eval_input",
    )

    inputs = [rec_input, eval_input]
    outputs = []

    if network_config := config.model["ConvLSTM"]:
        # Get ConvLSTM layers
        return_sequences = any((config.model["LSTM"], config.model["decoder"]))
        lstm_layers = conv_lstm_network(
            name="ConvLSTM",
            return_sequences=return_sequences,
            **network_config,
        )
        # Apply ConvLSTM layers
        rec_output = next(lstm_layers)(rec_input)
        for layer in lstm_layers:
            rec_output = layer(rec_output)

        # Reshape for next layer
        if return_sequences:
            new_shape = (-1, np.product(rec_output.shape[2:]))
            reshape_layer = keras.layers.Reshape(new_shape, name="reshape")
            rec_output = reshape_layer(rec_output)
        else:
            flatten_layer = keras.layers.Flatten(name="conv_flatten")
            last_output = flatten_layer(rec_output)
    else:
        rec_output = rec_input

    if encoder_config := config.model["encoder"]:
        rec_network = dense_network(name="rec_encoder", **encoder_config["rec"])
        for layer in rec_network:
            rec_output = layer(rec_output)

        eval_network = dense_network(name="eval_encoder", **encoder_config["eval"])
        eval_output = next(eval_network)(eval_input)
        for layer in eval_network:
            eval_output = layer(eval_output)
    else:
        eval_output = eval_input

    if network_config := config.model["LSTM"]:
        # Get LSTM layers
        return_sequences = config.model["decoder"] is not None
        lstm_layers = lstm_network(
            name="LSTM",
            return_sequences=return_sequences,
            **network_config,
        )
        # Apply ConvLSTM layers
        for layer in lstm_layers:
            rec_output = layer(rec_output)
        if return_sequences:
            lambda_layer = keras.layers.Lambda(lambda x: x[:, -1], name="last_seq")
            last_output = lambda_layer(rec_output)
        else:
            last_output = rec_output

    if decoder_config := config.model["decoder"]:
        crop_layer = keras.layers.Cropping1D(cropping=(0, 1), name="crop")
        prev_outputs = crop_layer(rec_output)

        rec_network = dense_network(name="rec_decoder", **decoder_config["rec"])
        rec_predictions = next(rec_network)(prev_outputs)
        for layer in rec_network:
            rec_predictions = layer(rec_predictions)

        flatten_layer = keras.layers.Flatten(name="rec_flatten")
        rec_predictions = flatten_layer(rec_predictions)

        eval_network = dense_network(name="eval_decoder", **decoder_config["eval"])
        eval_prediction = next(eval_network)(last_output)
        for layer in eval_network:
            eval_prediction = layer(eval_prediction)

        concat_layer = keras.layers.Concatenate(axis=1, name="predictions")
        predictions = concat_layer((rec_predictions, eval_prediction))

        outputs.append(predictions)

    concat_layer = keras.layers.Concatenate(axis=1, name="eval_concat")
    main_input = concat_layer((last_output, eval_output))

    eval_config = config.model["eval"]
    main_network = dense_network(
        name="main_eval",
        output_name="main_output",
        **eval_config["main"],
    )
    main_output = next(main_network)(main_input)
    for layer in main_network:
        main_output = layer(main_output)
    outputs.append(main_output)

    aux_network = dense_network(
        name="aux_eval",
        output_name="aux_output",
        **eval_config["aux"],
    )
    aux_output = next(aux_network)(last_output)
    for layer in aux_network:
        aux_output = layer(aux_output)
    outputs.append(aux_output)

    # Compile model
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)

    if opt_params := config.train["optimizer"]:
        optimizer = keras.optimizers.Adam(**opt_params)
    else:
        optimizer = "adam"

    model.compile(
        optimizer=optimizer,
        loss=config.train["loss"],
        loss_weights=config.train["loss_weights"],
        metrics=config.train["metrics"],
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
    name: str,
    units: List[int],
    activation: Optional[str] = None,
    dropout_rates: Optional[List[float]] = None,
    return_sequences: bool = False,
):
    num_layers = len(units)
    if dropout_rates is None:
        dropout_rates = list(repeat(None, num_layers))

    if len(dropout_rates) != num_layers:
        raise ValueError(
            f"Mismatch between the number of LSTM layers ({num_layers})"
            "and the number of LSTM dropout rate after each layer."
        )

    inds = range(1, num_layers + 1)
    for ind, layer_units, rate in zip(inds, units, dropout_rates):
        return_layer_sequences = (ind != num_layers) or return_sequences
        lstm_layer = keras.layers.LSTM(
            units=layer_units,
            return_sequences=return_layer_sequences,
            name=f"{name}{ind}",
        )
        yield lstm_layer

        if rate:
            dropout_layer = keras.layers.Dropout(
                rate=rate,
                name=f"dropout_{name}{ind}",
            )
            yield dropout_layer

    if activation:
        activation_layer = keras.layers.Activation(
            activation=activation,
            name=f"{activation}_{name}",
        )
        yield activation_layer


def conv_lstm_network(
    name: str,
    filters: List[int],
    kernel_sizes: List[int],
    activation: Optional[str] = None,
    dropout_rates: Optional[List[float]] = None,
    return_sequences: bool = False,
):
    num_layers = len(filters)
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
    for layer_params in zip(inds, filters, dropout_rates, kernel_sizes):
        ind, layer_filters, rate, size = layer_params
        return_layer_sequences = (ind != num_layers) or return_sequences
        convlstm_layer = keras.layers.ConvLSTM2D(
            filters=layer_filters,
            kernel_size=size,
            data_format="channels_first",
            return_sequences=return_layer_sequences,
            name=f"{name}{ind}",
        )
        yield convlstm_layer

        if rate:
            dropout_layer = keras.layers.Dropout(
                rate=rate,
                name=f"dropout_{name}{ind}",
            )
            yield dropout_layer

    if activation:
        activation_layer = keras.layers.Activation(
            activation=activation,
            name=f"{activation}_{name}",
        )
        yield activation_layer


def dense_network(
    name: str,
    units: List[int],
    activation: str,
    dropout_rates: Optional[List[float]] = None,
    l2_factor: Optional[float] = None,
    output_name: Optional[str] = None,
    final_activation: Optional[str] = None,
):
    num_layers = len(units)

    if dropout_rates is None:
        dropout_rates = list(repeat(None, num_layers))

    if len(dropout_rates) != num_layers:
        raise ValueError(
            f"Mismatch between the number of ConvLSTM layers ({num_layers})"
            "and the number of ConvLSTM dropout rate after each layer."
        )

    inds = range(1, num_layers + 1)
    for ind, layer_units, rate in zip(inds, units, dropout_rates):
        kernel_regularizer = keras.regularizers.L2(l2_factor) if l2_factor else None
        if ind == num_layers:
            layer_activation = final_activation or activation
            layer_name = output_name or f"{name}-{ind}"
        else:
            layer_activation = activation
            layer_name = f"{name}-{ind}"

        dense_layer = keras.layers.Dense(
            units=layer_units,
            activation=layer_activation,
            kernel_regularizer=kernel_regularizer,
            name=layer_name,
        )
        yield dense_layer

        if rate is not None:
            dropout_layer = keras.layers.Dropout(
                rate=rate,
                name=f"dropout_{layer_name}",
            )
            yield dropout_layer
