"""The 2-layer LSTM RNN model use for the decoder."""
from typing import Callable, Dict, Optional, Tuple, Union

# from tensorflow import keras
import keras
import tensorflow as tf

from ..utils.config import Config


def get_model(
    defects_shape: Tuple[int, ...],
    final_defects_shape: Tuple[int, ...],
    config: Config,
    optimizer: Optional[str] = None,
    loss: Optional[Dict[str, Union[str, Callable]]] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    metrics: Optional[Dict[str, str]] = None,
    name: Optional[str] = None,
) -> keras.Model:
    """
    get_model _summary_

    Parameters
    ----------
    syn_shape : Tuple[int, ...]
        The shape of the syndrome defects that the main head of the model takes as input.
        The shape is expected to be a tuple of the order (number of QEC rounds, number of ancilla qubits).
        In the case that the number of QEC rounds is a variable, None should be given instead.
    final_syn_shape : Tuple[int, ...]
        The shape of either the final (projected) syndrome defects or the final data qubit measurement
        that only the main head of the model takes as input.
        The shape is expected to be the tuple (number of ancilla qubits, ).
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
    rate = config.train.get("dropout_rate")

    defects = keras.layers.Input(
        shape=defects_shape,
        dtype="float32",
        name="defects",
    )

    final_defects = keras.layers.Input(
        shape=final_defects_shape,
        dtype="float32",
        name="final_defects",
    )

    lstm_layer = keras.layers.LSTM(
        config.model.get("LSTM_units", 64),
        return_sequences=True,
        name="LSTM_1",
    )
    output = lstm_layer(defects)

    if rate is not None:
        dropout_layer = keras.layers.Dropout(
            rate=rate,
            name="dropout_LSTM_1",
        )
        output = dropout_layer(output)

    lstm_layer = keras.layers.LSTM(
        config.model.get("LSTM_units", 64),
        return_sequences=False,
        name="LSTM_2",
    )
    output = lstm_layer(output)

    if rate is not None:
        dropout_layer = keras.layers.Dropout(
            rate=rate,
            name="dropout_LSTM_2",
        )
        output = dropout_layer(output)

    act_layer = keras.layers.Activation(
        activation="relu",
        name="relu",
    )
    output = act_layer(output)

    concat_input = tf.concat((output, final_defects), axis=1)

    regulizar = keras.regularizers.L2(l2=config.train["l2_factor"])
    dense_layer = keras.layers.Dense(
        config.model["eval_units"],
        activation="relu",
        kernel_regularizer=regulizar,
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
        config.model["output_units"],
        activation="sigmoid",
        name="main_output",
    )
    main_output = output_layer(main_output)

    regulizar = keras.regularizers.L2(l2=config.train["l2_factor"])
    dense_layer = keras.layers.Dense(
        config.model["eval_units"],
        activation="relu",
        kernel_regularizer=regulizar,
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
        config.model["output_units"],
        activation="sigmoid",
        name="aux_output",
    )
    aux_output = dense_layer(aux_output)

    model = keras.Model(
        inputs=[defects, final_defects],
        outputs=[main_output, aux_output],
        name=name or "decoder_model",
    )

    if optimizer is None:
        opt_param = config.train.get("optimizer", "adam")
        if isinstance(opt_param, dict):
            optimizer = keras.optimizers.Adam(**opt_param)
        else:
            optimizer = opt_param

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

    return model
