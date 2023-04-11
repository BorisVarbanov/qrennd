from typing import List, Union

import numpy as np
from tensorflow import keras

from ..configs import Config
from .networks import (
    conv_lstm_network,
    decoder_network,
    eval_network,
    lstm_network,
    conv_network,
)

DEFAULT_OPT_PARAMS = dict(learning_rate=0.001)


def lstm_model(rec_features: int, eval_features: int, config: Config) -> keras.Model:
    rec_input = keras.layers.Input(
        shape=(None, rec_features),
        dtype="float32",
        name="rec_input",
    )
    eval_input = keras.layers.Input(
        shape=(eval_features,),
        dtype="float32",
        name="eval_input",
    )

    lstm_params = config.model["LSTM"]
    network = lstm_network(name="LSTM", **lstm_params)
    output = next(network)(rec_input)
    for layer in network:
        output = layer(output)

    activation_layer = keras.layers.Activation(
        activation="relu",
        name="relu_LSTM",
    )
    output = activation_layer(output)

    concat_layer = keras.layers.Concatenate(axis=1, name="eval_concat")
    main_eval_input = concat_layer((output, eval_input))

    main_eval_params = config.model["main_eval"]
    network = eval_network(name="main", **main_eval_params)
    main_output = next(network)(main_eval_input)
    for layer in network:
        main_output = layer(main_output)

    aux_eval_params = config.model["aux_eval"]
    network = eval_network(name="aux", **aux_eval_params)
    aux_output = next(network)(output)
    for layer in network:
        aux_output = layer(aux_output)

    inputs = [rec_input, eval_input]
    outputs = [main_output, aux_output]

    model = keras.Model(inputs=inputs, outputs=outputs, name="lstm_model")

    opt_params = config.train.get("optimizer", DEFAULT_OPT_PARAMS)
    optimizer = keras.optimizers.Adam(**opt_params)

    loss = config.train.get("loss")
    loss_weights = config.train.get("loss_weights")
    metrics = config.train.get("metrics")

    model.compile(optimizer, loss, metrics, loss_weights)

    if config.init_weights:
        try:
            experiment_dir = config.output_dir / config.experiment
            model.load_weights(experiment_dir / config.init_weights)
        except FileNotFoundError as error:
            raise ValueError(
                "Invalid initial weights in configuration file."
            ) from error

    return model


def convlstm_model(
    rec_features: List[int], eval_features: int, config: Config
) -> keras.Model:
    rec_input = keras.layers.Input(
        shape=(None, *rec_features),
        dtype="float32",
        name="rec_input",
    )
    eval_input = keras.layers.Input(
        shape=(eval_features,),
        dtype="float32",
        name="eval_input",
    )

    conv_lstm_params = config.model["ConvLSTM"]
    return_sequences = config.model["LSTM"] is not None
    network = conv_lstm_network(
        name="ConvLSTM",
        return_sequences=return_sequences,
        **conv_lstm_params,
    )
    output = next(network)(rec_input)
    for layer in network:
        output = layer(output)

    activation_layer = keras.layers.Activation(
        activation="relu",
        name="relu_ConvLSTM",
    )
    output = activation_layer(output)

    if lstm_params := config.model["LSTM"]:
        new_shape = (-1, np.product(output.shape[2:]))
        reshape_layer = keras.layers.Reshape(new_shape, name="conv_reshape")
        output = reshape_layer(output)

        network = lstm_network(name="LSTM", **lstm_params)
        for layer in network:
            output = layer(output)

        activation_layer = keras.layers.Activation(
            activation="relu",
            name="relu_LSTM",
        )
        output = activation_layer(output)
    else:
        flatten_layer = keras.layers.Flatten(
            name="flatten_output", data_format="channels_first"
        )
        output = flatten_layer(output)

    concat_layer = keras.layers.Concatenate(axis=1, name="eval_concat")
    main_eval_input = concat_layer((output, eval_input))

    main_eval_params = config.model["main_eval"]
    network = eval_network(name="main", **main_eval_params)
    main_output = next(network)(main_eval_input)
    for layer in network:
        main_output = layer(main_output)

    aux_eval_params = config.model["aux_eval"]
    network = eval_network(name="aux", **aux_eval_params)
    aux_output = next(network)(output)
    for layer in network:
        aux_output = layer(aux_output)

    inputs = [rec_input, eval_input]
    outputs = [main_output, aux_output]

    model = keras.Model(inputs=inputs, outputs=outputs, name="convlstm_model")

    opt_params = config.train.get("optimizer", DEFAULT_OPT_PARAMS)
    optimizer = keras.optimizers.Adam(**opt_params)

    loss = config.train.get("loss")
    loss_weights = config.train.get("loss_weights")
    metrics = config.train.get("metrics")

    model.compile(optimizer, loss, metrics, loss_weights)

    if config.init_weights:
        try:
            experiment_dir = config.output_dir / config.experiment
            model.load_weights(experiment_dir / config.init_weights)
        except FileNotFoundError as error:
            raise ValueError(
                "Invalid initial weights in configuration file."
            ) from error

    return model


def conv_lstm_model(
    rec_features: List[int], eval_features: int, config: Config
) -> keras.Model:
    rec_input = keras.layers.Input(
        shape=(None, *rec_features),
        dtype="float32",
        name="rec_input",
    )
    eval_input = keras.layers.Input(
        shape=(eval_features,),
        dtype="float32",
        name="eval_input",
    )

    conv_params = config.model["Conv"]
    network = conv_network(
        name="Conv",
        **conv_params,
    )
    output = next(network)(rec_input)
    for layer in network:
        output = layer(output)

    activation_layer = keras.layers.Activation(
        activation="relu",
        name="relu_Conv",
    )
    output = activation_layer(output)

    new_shape = (-1, np.product(output.shape[2:]))
    reshape_layer = keras.layers.Reshape(new_shape, name="conv_reshape")
    output = reshape_layer(output)

    lstm_params = config.model["LSTM"]
    network = lstm_network(name="LSTM", **lstm_params)
    for layer in network:
        output = layer(output)

    activation_layer = keras.layers.Activation(
        activation="relu",
        name="relu_LSTM",
    )
    output = activation_layer(output)

    concat_layer = keras.layers.Concatenate(axis=1, name="eval_concat")
    main_eval_input = concat_layer((output, eval_input))

    main_eval_params = config.model["main_eval"]
    network = eval_network(name="main", **main_eval_params)
    main_output = next(network)(main_eval_input)
    for layer in network:
        main_output = layer(main_output)

    aux_eval_params = config.model["aux_eval"]
    network = eval_network(name="aux", **aux_eval_params)
    aux_output = next(network)(output)
    for layer in network:
        aux_output = layer(aux_output)

    inputs = [rec_input, eval_input]
    outputs = [main_output, aux_output]

    model = keras.Model(inputs=inputs, outputs=outputs, name="conv_lstm_model")

    opt_params = config.train.get("optimizer", DEFAULT_OPT_PARAMS)
    optimizer = keras.optimizers.Adam(**opt_params)

    loss = config.train.get("loss")
    loss_weights = config.train.get("loss_weights")
    metrics = config.train.get("metrics")

    model.compile(optimizer, loss, metrics, loss_weights)

    if config.init_weights:
        try:
            experiment_dir = config.output_dir / config.experiment
            model.load_weights(experiment_dir / config.init_weights)
        except FileNotFoundError as error:
            raise ValueError(
                "Invalid initial weights in configuration file."
            ) from error

    return model


def lstm_decoder_model(
    rec_features: int, eval_features: int, config: Config
) -> keras.Model:
    rec_input = keras.layers.Input(
        shape=(None, rec_features),
        dtype="float32",
        name="rec_input",
    )
    eval_input = keras.layers.Input(
        shape=(eval_features,),
        dtype="float32",
        name="eval_input",
    )

    lstm_params = config.model["LSTM"]
    network = lstm_network(name="LSTM", return_sequences=True, **lstm_params)
    outputs = next(network)(rec_input)
    for layer in network:
        outputs = layer(outputs)

    lambda_layer = keras.layers.Lambda(lambda x: x[:, -1], name="last_sequence")
    last_output = lambda_layer(outputs)

    crop_layer = keras.layers.Cropping1D(cropping=(0, 1), name="crop")
    prev_outputs = crop_layer(outputs)

    rec_decoder_params = config.model["rec_decoder"]
    network = decoder_network(name="main", **rec_decoder_params)
    rec_predictions = next(network)(prev_outputs)
    for layer in network:
        rec_predictions = layer(rec_predictions)

    flatten_layer = keras.layers.Flatten(name="flatten_predictions")
    rec_predictions = flatten_layer(rec_predictions)

    eval_decoder_params = config.model["eval_decoder"]
    network = decoder_network(name="eval", **eval_decoder_params)
    eval_prediction = next(network)(last_output)
    for layer in network:
        eval_prediction = layer(eval_prediction)

    concat_layer = keras.layers.Concatenate(axis=1, name="predictions")
    predictions = concat_layer((rec_predictions, eval_prediction))

    concat_layer = keras.layers.Concatenate(axis=1, name="eval_concat")
    main_eval_input = concat_layer((last_output, eval_input))

    main_eval_params = config.model["main_eval"]
    network = eval_network(name="main", **main_eval_params)
    main_output = next(network)(main_eval_input)
    for layer in network:
        main_output = layer(main_output)

    aux_eval_params = config.model["aux_eval"]
    network = eval_network(name="aux", **aux_eval_params)
    aux_output = next(network)(last_output)
    for layer in network:
        aux_output = layer(aux_output)

    inputs = [rec_input, eval_input]
    outputs = [main_output, aux_output, predictions]

    model = keras.Model(inputs=inputs, outputs=outputs, name="lstm_decoder_model")

    opt_params = config.train.get("optimizer", DEFAULT_OPT_PARAMS)
    optimizer = keras.optimizers.Adam(**opt_params)

    loss = config.train.get("loss")
    loss_weights = config.train.get("loss_weights")
    metrics = config.train.get("metrics")

    model.compile(optimizer, loss, metrics, loss_weights)

    if config.init_weights:
        try:
            experiment_dir = config.output_dir / config.experiment
            model.load_weights(experiment_dir / config.init_weights)
        except FileNotFoundError as error:
            raise ValueError(
                "Invalid initial weights in configuration file."
            ) from error

    return model


def get_model(
    rec_features: Union[int, List[int]],
    eval_features: int,
    config: Config,
) -> keras.Model:
    model_type = config.model["type"]
    if model_type == "LSTM":
        return lstm_model(rec_features, eval_features, config)
    elif model_type == "ConvLSTM":
        return convlstm_model(rec_features, eval_features, config)
    elif model_type == "Conv+LSTM":
        return conv_lstm_model(rec_features, eval_features, config)
    elif model_type == "LSTM_decoder":
        return lstm_decoder_model(rec_features, eval_features, config)
    else:
        raise ValueError(f"Unrecognize config.model.type {model_type}")
