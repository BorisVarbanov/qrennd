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
        name="recurrent_input",
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
            return_sequences=to_LSTM_input,
        )
        # Apply ConvLSTM layers
        conv_output = next(convlstm_layers)(recurrent_input)
        for layer in convlstm_layers:
            conv_output = layer(conv_output)

        # Reshape for next layer
        if to_LSTM_input:
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
    units: List[int],
    dropout_rates: Optional[List[float]] = None,
):
    num_layers = len(units)
    if dropout_rates is None:
        dropout_rates = list(repeat(None, len(units)))

    if len(dropout_rates) != num_layers:
        raise ValueError(
            f"Mismatch between the number of LSTM layers ({num_layers})"
            "and the number of LSTM dropout rate after each layer."
        )
    inds = range(1, num_layers + 1)

    for ind_, units_, rate_ in zip(inds, units, dropout_rates):
        return_sequences_ = ind_ != num_layers
        lstm_layer = keras.layers.LSTM(
            units=units_,
            return_sequences=return_sequences_,
            name=f"LSTM_{ind_}",
        )
        yield lstm_layer

        if rate_ is not None:
            dropout_layer = keras.layers.Dropout(
                rate=rate_,
                name=f"dropout_LSTM_{ind_}",
            )
            yield dropout_layer

    act_layer = keras.layers.Activation(
        activation="relu",
        name="relu_lstm",
    )
    yield act_layer


def conv_lstm_network(
    filters: List[int],
    kernel_sizes: List[int],
    dropout_rates: Optional[List[float]] = None,
    return_sequences: Optional[bool] = False,
):
    num_layers = len(filters)
    if dropout_rates is None:
        dropout_rates = list(repeat(None, num_layers))

    if len(dropout_rates) != num_layers:
        raise ValueError(
            f"Mismatch between the number of ConvLSTM layers ({num_layers})"
            "and the number of ConvLSTM dropout rate after each layer."
        )
    if len(kernels) != num_layers:
        raise ValueError(
            f"Mismatch between the number of ConvLSTM layers ({num_layers})"
            "and the number of ConvLSTM kernel sizes."
        )

    inds = range(1, num_layers + 1)
    for ind_, filters_, rate_, kernel_size_ in zip(
        inds, filters, dropout_rates, kernel_sizes
    ):
        return_sequences_ = (ind_ != num_layers) or return_sequences
        convlstm_layer = keras.layers.ConvLSTM2D(
            filters=filters_,
            kernel_size=kernel_size_,
            data_format="channels_first",
            return_sequences=return_sequences_,
            name=f"ConvLSTM_{ind_}",
        )
        yield convlstm_layer

        if rate_ is not None:
            dropout_layer = keras.layers.Dropout(
                rate=rate_,
                name=f"dropout_ConvLSTM_{ind_}",
            )
            yield dropout_layer

    act_layer = keras.layers.Activation(
        activation="relu",
        name="relu_convlstm",
    )
    yield act_layer


def evaluation_network(
    units: List[int],
    dropout_rates: Optional[List[float]] = None,
    l2_factor: Optional[float] = None,
    name: Optional[str] = "eval",
):
    num_layers = len(units)
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

    inds = range(1, num_layers)
    for ind_, units_, rate_, kernel_size_ in zip(
        inds, units, dropout_rates, kernel_sizes
    ):
        dense_layer = keras.layers.Dense(
            units=units_,
            activation=activation_,
            kernel_regularizer=regularizer,
            name=f"{name}_dense_{ind_}",
        )
        yield dense_layer

        if rate_ is not None:
            dropout_layer = keras.layers.Dropout(
                rate=rate_,
                name=f"dropout_{name}_dense_{ind_}",
            )
            yield dropout_layer

    # Final evaluation layer
    output_layer = keras.layers.Dense(
        units=units[-1],
        activation="sigmoid",
        name=f"{name}_output",
    )
    yield output_layer
