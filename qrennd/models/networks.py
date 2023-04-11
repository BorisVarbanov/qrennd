from itertools import repeat
from typing import Iterator, List, Optional, Union

from tensorflow import keras

opt_float = Union[float, None]

Network = Iterator[keras.layers.Layer]


def lstm_network(
    name: str,
    units: List[int],
    dropout_rates: Optional[List[opt_float]] = None,
    return_sequences: bool = False,
) -> Network:
    num_layers = len(units)
    dropout_rates = dropout_rates or list(repeat(None, num_layers))

    if len(dropout_rates) != num_layers:
        raise ValueError(
            f"Mismatch between the number of LSTM layers ({num_layers})"
            "and the number of LSTM dropout rate after each layer."
        )

    inds = range(1, num_layers + 1)
    for ind, layer_units, rate in zip(inds, units, dropout_rates):
        layer_name = f"{name}-{ind}"
        return_seq = (ind != num_layers) or return_sequences

        lstm_layer = keras.layers.LSTM(
            units=layer_units, return_sequences=return_seq, name=layer_name
        )
        yield lstm_layer

        if rate:
            dropout_layer = keras.layers.Dropout(rate, name=f"dropout_{layer_name}")
            yield dropout_layer


def conv_lstm_network(
    name: str,
    filters: List[int],
    kernel_sizes: List[int],
    dropout_rates: Optional[List[opt_float]] = None,
    return_sequences: bool = False,
) -> Network:
    num_layers = len(filters)
    dropout_rates = dropout_rates or list(repeat(None, num_layers))

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
    for params in zip(inds, filters, dropout_rates, kernel_sizes):
        ind, layer_filters, rate, size = params

        layer_name = f"{name}-{ind}"
        return_seq = (ind != num_layers) or return_sequences

        lstm_layer = keras.layers.ConvLSTM2D(
            filters=layer_filters,
            kernel_size=size,
            return_sequences=return_seq,
            data_format="channels_first",
            name=layer_name,
        )
        yield lstm_layer

        if rate:
            dropout_layer = keras.layers.Dropout(rate, name=f"dropout_{layer_name}")
            yield dropout_layer


def conv_network(
    name: str,
    filters: List[int],
    kernel_sizes: List[int],
    dropout_rates: Optional[List[opt_float]] = None,
) -> Network:
    num_layers = len(filters)
    dropout_rates = dropout_rates or list(repeat(None, num_layers))

    if len(dropout_rates) != num_layers:
        raise ValueError(
            f"Mismatch between the number of Conv layers ({num_layers})"
            "and the number of Conv dropout rate after each layer."
        )
    if len(kernel_sizes) != num_layers:
        raise ValueError(
            f"Mismatch between the number of Conv layers ({num_layers})"
            "and the number of Conv kernel sizes."
        )

    inds = range(1, num_layers + 1)
    for params in zip(inds, filters, dropout_rates, kernel_sizes):
        ind, layer_filters, rate, size = params

        layer_name = f"{name}-{ind}"

        conv_layer = keras.layers.Conv2D(
            filters=layer_filters,
            kernel_size=size,
            data_format="channels_first",
            name=layer_name,
        )
        conv_layer = keras.layers.TimeDistributed(conv_layer)
        yield conv_layer

        if rate:
            dropout_layer = keras.layers.Dropout(rate, name=f"dropout_{layer_name}")
            yield dropout_layer


def eval_network(
    name: str,
    units: List[int],
    dropout_rates: Optional[List[opt_float]] = None,
    l2_factor: Optional[float] = None,
    *,
    activation: str = "relu",
    output_activation: str = "sigmoid",
) -> Network:
    num_layers = len(units)
    if dropout_rates is None:
        dropout_rates = list(repeat(None, num_layers))

    if len(dropout_rates) != num_layers:
        raise ValueError(
            f"Mismatch between the number of ConvLSTM layers ({num_layers})"
            "and the number of ConvLSTM dropout rate after each layer."
        )

    regularizer = keras.regularizers.L2(l2_factor) if l2_factor else None

    inds = range(1, num_layers + 1)
    for ind, layer_units, rate in zip(inds, units, dropout_rates):
        is_output = ind == num_layers
        activation = output_activation if is_output else activation
        layer_name = f"{name}_output" if is_output else f"{name}_eval-{ind}"

        dense_layer = keras.layers.Dense(
            units=layer_units,
            activation=activation,
            kernel_regularizer=regularizer,
            name=layer_name,
        )
        yield dense_layer

        if rate and not is_output:
            dropout_layer = keras.layers.Dropout(rate, name=f"dropout_{layer_name}")
            yield dropout_layer


def decoder_network(
    name: str,
    units: List[int],
    dropout_rates: Optional[List[opt_float]] = None,
    l2_factor: Optional[float] = None,
    *,
    activation: str = "relu",
    output_activation: str = "sigmoid",
) -> Network:
    num_layers = len(units)
    if dropout_rates is None:
        dropout_rates = list(repeat(None, num_layers))

    if len(dropout_rates) != num_layers:
        raise ValueError(
            f"Mismatch between the number of ConvLSTM layers ({num_layers})"
            "and the number of ConvLSTM dropout rate after each layer."
        )

    regularizer = keras.regularizers.L2(l2_factor) if l2_factor else None

    inds = range(1, num_layers + 1)
    for ind, layer_units, rate in zip(inds, units, dropout_rates):
        is_output = ind == num_layers
        activation = output_activation if is_output else activation
        layer_name = f"{name}_prediction" if is_output else f"{name}_dec-{ind}"

        dense_layer = keras.layers.Dense(
            units=layer_units,
            activation=activation,
            kernel_regularizer=regularizer,
            name=layer_name,
        )
        yield dense_layer

        if rate and not is_output:
            dropout_layer = keras.layers.Dropout(rate, name=f"dropout_{layer_name}")
            yield dropout_layer
