"""The 2-layer LSTM RNN model use for the decoder."""
from typing import Optional, Tuple, Union

import tensorflow as tf
from tensorflow import keras

from ..utils.config import Config

layers = keras.layers
optimizers = keras.optimizers


def get_model(
    syn_shape: Tuple[Union[int, None], int],
    proj_syn_shape: Tuple[int],
    config: Config,
    *,
    name: Optional[str] = None
) -> keras.Model:
    """
    get_model Build and compile the recurrent neural network model.

    Parameters
    ----------
    syn_shape : Tuple[Union[int, None], int]
        The shape of the syndrome defects that the model takes as input.
        The shape is expected to be a tuple of the order (number of QEC rounds, number of ancilla qubits).
        In the case that the number of QEC rounds is a variable, None should be given instead.
    proj_syn_shape : Tuple[int]
        The shape of the final (projected) syndrome defects that the main head of the model takes as input.
        The shape is expected to be the tuple (number of ancilla qubits, ).
    config : Config
        The model configuartion, given as a Config dataclass. The configuation file should define the model property with following keys:
            lstm_units - the number of units each LSTM layer uses.
            eval_units - number of units in the hidden evaluation layer.
            output_units - the number of units in the output layer (for binary classification this should be 1).
        And a train dictiornary with the following keys:
            learning_rate - the learning rate of the Adam optimizer.
            dropout_rate - the dropout rate used after each LSTM layer and the hidden evaluation layer.
            l2_factor - the weight decay factor used to regularize the weights of the hidden evaluation layer.
    name : Optional[str], optional
        The name of the model, by default None

    Returns
    -------
    tensorflow.keras.Model
        The build and compiled model.
    """

    defects = keras.layers.Input(
        syn_shape,
        dtype="float32",
        name="Defects",
    )
    final_defects = keras.layers.Input(
        shape=proj_syn_shape,
        dtype="float32",
        name="Final defects",
    )

    lstm_layer = layers.LSTM(
        config.model["lstm_units"], return_sequences=True, name="LSTM_1"
    )
    output = lstm_layer(defects)

    dropout_layer = layers.Dropout(
        rate=config.train["dropout_rate"], name="dropout_LSTM_1"
    )
    output = dropout_layer(output)

    lstm_layer = layers.LSTM(
        config.model["lstm_units"], return_sequences=False, name="LSTM_2"
    )
    output = lstm_layer(output)

    dropout_layer = layers.Dropout(
        rate=config.train["dropout_rate"], name="dropout_LSTM_2"
    )
    output = dropout_layer(output)

    act_layer = layers.Activation("relu", name="relu")
    output = act_layer(output)

    concat_input = tf.concat((output, final_defects), axis=1)

    regulizar = keras.regularizers.L2(config.train["l2_factor"])
    dense_layer = layers.Dense(
        config.model["eval_units"],
        activation="relu",
        kernel_regularizer=regulizar,
        name="main_dense",
    )
    main_output = dense_layer(concat_input)

    dropout_layer = layers.Dropout(
        rate=config.train["dropout_rate"], name="dropout_main_dense"
    )
    main_output = dropout_layer(main_output)

    output_layer = layers.Dense(
        config.model["output_units"], activation="sigmoid", name="main_output"
    )
    main_output = output_layer(main_output)

    regulizar = keras.regularizers.L2(config.train["l2_factor"])
    dense_layer = layers.Dense(
        config.model["eval_units"],
        activation="relu",
        kernel_regularizer=regulizar,
        name="aux_dense",
    )
    aux_output = dense_layer(output)

    dropout_layer = layers.Dropout(
        rate=config.train["dropout_rate"], name="dropout_aux_dense"
    )
    aux_output = dropout_layer(aux_output)

    dense_layer = layers.Dense(
        config.model["output_units"], activation="sigmoid", name="aux_output"
    )
    aux_output = dense_layer(aux_output)

    model = keras.Model(
        inputs=[defects, final_defects],
        outputs=[main_output, aux_output],
        name=name or "Decoder model",
    )
    optimizer = optimizers.Adam(
        learning_rate=config.train["optimizer"]["learning_rate"]
    )

    model.compile(
        optimizer=optimizer,
        loss=config.train["losses"],
        loss_weights=config.train["loss_weights"],
        metrics=config.train["metrics"],
    )

    return model
