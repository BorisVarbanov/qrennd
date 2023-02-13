"""The 2-layer LSTM RNN model use for the decoder."""
from itertools import repeat
from typing import Callable, Dict, Optional, Union

from tensorflow import concat, keras

from ..configs import Config


def get_model(
    seq_size: int,
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
    seq_size : int
        The size of each sequence that is given to the LSTM layers.
    vec_size : Tuple[int]
        The size of the vector given directly to the evaluation layer.
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
    lstm_input = keras.layers.Input(
        shape=(None, seq_size),
        dtype="float32",
        name="lstm_input",
    )

    eval_input = keras.layers.Input(
        shape=(vec_size,),
        dtype="float32",
        name="eval_input",
    )

    lstm_units = config.model["LSTM_units"]
    dropout_rates = config.model.get("LSTM_dropout_rates")

    num_layers = len(lstm_units)
    if dropout_rates is None:
        dropout_rates = repeat(None, len(lstm_units))

    if len(dropout_rates) != num_layers:
        raise ValueError(
            f"Mismatch between the number of LSTM layers ({num_layers})"
            "and the number of LSTM dropout rate after each layer."
        )
    inds = range(1, num_layers + 1)

    output = None
    for ind, units, rate in zip(inds, lstm_units, dropout_rates):
        return_sequences = ind != num_layers
        lstm_layer = keras.layers.LSTM(
            units=units,
            return_sequences=return_sequences,
            name=f"LSTM_{ind}",
        )
        if output is None:
            output = lstm_layer(lstm_input)
        else:
            output = lstm_layer(output)

        if rate is not None:
            dropout_layer = keras.layers.Dropout(
                rate=rate,
                name=f"dropout_LSTM_{ind}",
            )
            output = dropout_layer(output)

    act_layer = keras.layers.Activation(
        activation="relu",
        name="relu",
    )
    output = act_layer(output)

    concat_input = concat((output, eval_input), axis=1)

    l2_factor = config.model.get("l2_factor")
    if l2_factor is not None:
        regularizer = keras.regularizers.L2(l2_factor)
    else:
        regularizer = None

    eval_units = config.model.get("eval_units", 64)

    rate = config.model.get("eval_dropout_rate")
    output_units = config.model.get("output_units", 1)

    dense_layer = keras.layers.Dense(
        units=eval_units,
        activation="relu",
        kernel_regularizer=regularizer,
        name="main_dense",
    )
    main_output = dense_layer(concat_input)

    if rate is not None:
        dropout_layer = keras.layers.Dropout(
            rate=rate,
            name="dropout_main_dense",
        )
        main_output = dropout_layer(main_output)

    output_layer = keras.layers.Dense(
        units=output_units,
        activation="sigmoid",
        name="main_output",
    )
    main_output = output_layer(main_output)

    if l2_factor is not None:
        regularizer = keras.regularizers.L2(l2_factor)
    else:
        regularizer = None

    dense_layer = keras.layers.Dense(
        units=eval_units,
        activation="relu",
        kernel_regularizer=regularizer,
        name="aux_dense",
    )
    aux_output = dense_layer(output)

    if rate is not None:
        dropout_layer = keras.layers.Dropout(
            rate=rate,
            name="dropout_aux_dense",
        )
        aux_output = dropout_layer(aux_output)

    dense_layer = keras.layers.Dense(
        units=output_units,
        activation="sigmoid",
        name="aux_output",
    )
    aux_output = dense_layer(aux_output)

    model = keras.Model(
        inputs=[lstm_input, eval_input],
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
