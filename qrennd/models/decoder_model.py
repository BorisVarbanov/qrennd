"""The 2-layer LSTM RNN model use for the decoder"""
from typing import final

from tensorflow.python.keras.layers import (
    LSTM,
    Activation,
    Concatenate,
    Dense,
    Dropout,
    Input,
)
from tensorflow.python.keras.models import Model

from ..utils.config import Config
from .base_model import BaseModel


class DecoderModel(BaseModel):
    def __init__(self, config: Config) -> None:
        """
        __init__ Initialize the decoder model.

        Parameters
        ----------
        config : Config
            The Config object containing the model hyperparameters.
        """
        super().__init__(config)

    def build(
        self,
        defects_dim: int,
        final_defects_dim: int,
        lstm_layer_size: int,
        dense_layer_dims: int,
        dropout_rate: float,
    ) -> None:
        """Build the RNN model."""
        defects = Input(
            shape=(None, defects_dim),
            dtype="float32",
            name="defects",
        )
        final_defects = Input(
            shape=(None, final_defects_dim),
            dtype="float32",
            name="defects",
        )

        output = LSTM(lstm_layer_size, return_sequences=True)(defects)
        output = Dropout(rate=dropout_rate)(output)

        output = LSTM(lstm_layer_size, return_sequences=False)(output)
        output = Dropout(rate=dropout_rate)(output)
        output = Activation("relu")(output)

        aux_output = Dense(dense_layer_dims, activation="relu")(output)
        aux_output = Dense(dense_layer_dims, activation="relu")(aux_output)
        aux_output = Dense(1, activation="linear")(aux_output)

        main_input = Concatenate(axis=1)([output, final_defects])
        main_output = Dense(dense_layer_dims + final_defects_dim, activation="relu")(
            main_input
        )
        main_output = Dense(dense_layer_dims, activation="relu")(main_output)
        main_output = Dense(1, activation="linear")(main_output)

        self.model = Model(
            inputs=[defects, final_defects],
            outputs=[main_output, aux_output],
            name="qrennd_decoder",
        )
