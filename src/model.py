"""Contains the Tensorflow Seq2Seq model and layers to be used as a predictive
model for time series forecasting.

Copyright (C) 2022  KaidenFrizu

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""
from typing import Optional
from datetime import datetime
import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    """A custom Tensorflow layer dedicated for the encoding phase of the
    Seq2Seq model.

    Args:
        input_length: The number of data points over time to be fed as input.
        n_features: The number of features to be fed as past inputs.
        units: The number of LSTM units to be created on LSTM encoder.

    Attributes:
        input_layer: A `tensorflow.keras.layers.InputLayer` that enforces the
            required input shape of the model, determined by `input_length`
            and `n_features` arguments.
        lstm: A bidirectional LSTM layer that is responsible for the encoding
            process of the model.
        init_resolve: A Dense layer distributed over time to create an
            initial prediction at the current timestep.
    """

    def __init__(
        self,
        input_length: int,
        n_features: int,
        units: int,
    ):

        super().__init__(name='Encoder')

        self.input_layer = tf.keras.layers.InputLayer(
            input_shape=(input_length, n_features),
            name='input_encoder',
        )
        core_lstm = tf.keras.layers.LSTM(
            units=units,
            activation='tanh',
            dropout=0.1,
            kernel_regularizer='l1',
            name='encoder_lstm',
            return_sequences=True,
        )
        self.lstm = tf.keras.layers.Bidirectional(
            layer=core_lstm,
            merge_mode='concat',
        )
        core_dense = tf.keras.layers.Dense(1)
        self.init_resolve = tf.keras.layers.TimeDistributed(core_dense)

    def call(self, x, training=None):
        """A method to call the model that acts as the __call__() method.

        See https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer

        Args:
            x: A tensor input
            training: Determines whether to behave the decoder under training
                mode. This would occur a dropout on the weights per epoch.

        Returns:
            A sequence of length (input_length, 1)
        """
        x = self.input_layer(x)
        x = self.lstm(x, training=training)
        return self.init_resolve(x)


class Decoder(tf.keras.layers.Layer):
    """A custom Tensorflow layer dedicated for the decoding phase of the
    Seq2Seq model.

    The `lstm` attribute is a wrapped version of `lstmcell` where individidual
    timestep prediction is used for the latter.

    Args:
        units: The number of LSTM units to be created on LSTM decoder.
        horizon: The number of future timesteps to cast a prediction.

    Attributes:
        lstmcell: A core LSTM cell for the decoding framework.
        lstm: A wrapped LSTM layer to infer future values.
        resolve: A Dense layer wrapped with `tf.keras.layers.TimeDistributed`
            to produce a series of outputs.
    """

    def __init__(
        self,
        units: int,
        horizon: int,
    ):
        super().__init__(name='Decoder')

        self.horizon = horizon

        self.lstmcell = tf.keras.layers.LSTMCell(
            units=units,
            activation='tanh',
            dropout=0.2,
            kernel_regularizer='l1',
            bias_regularizer='l2',
            name='AR_cell',
        )
        self.lstm = tf.keras.layers.LSTM(
            units=units,
            activation='tanh',
            dropout=0.2,
            kernel_regularizer='l1',
            bias_regularizer='l2',
            name='decoder_lstm',
            return_state=True,
        )
        self.resolve = tf.keras.layers.Dense(1)

    def call(self, x, training=None):
        """A method to call the model that acts as the __call__() method.

        See https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer

        Args:
            x: A tensor input
            training: Determines whether to behave the decoder under training
                mode. This would occur a dropout on the weights per epoch.

        Returns:
            A sequence of length (horizon, 1)
        """
        preds = tf.TensorArray(tf.float32, size=self.horizon)
        x, *states = self.lstm(x, training=training)

        for h in tf.range(self.horizon):
            x, states = self.lstmcell(x, states=states, training=training)
            x_resolve = self.resolve(x)
            preds = preds.write(h, x_resolve)

        preds = preds.stack()

        return tf.transpose(preds, perm=[1, 0, 2])


class BitcoinRNN(tf.keras.Model):
    """A Tensorflow RNN model with a sequence to sequence (Seq2Seq) framework
    to predict future Bitcoin prices given multiple time series features.

    This is subclassed from `tensorflow.keras.Model`. The following arguments
    and attributes stated here are those that are not present in the default
    `tensorflow.keras.Model` class. For more information, visit the following
    documentation.

    https://www.tensorflow.org/api_docs/python/tf/keras/Model

    Args:
        input_length: The number of data points over time to be fed as input.
        horizon: The number of data points from the end of input data point
            to cast a prediction from the model.
        n_features: The number of features to be fed as past inputs.
        model_name: The ame of the RNN model. An initial name is provided
            which indicates the time this class was instantiated.
        units: The number of LSTM units to be created on both LSTM encoder
            and decoder.
        **kwargs: Additional arguments passed on `tensorflow.keras.Model`.

    Attributes:
        input_length: The number of data points over time to be fed as input.
        horizon: The number of data points from the end of input data point
            to cast a prediction from the model.
        n_features: The number of features to be fed as past inputs.
        model_name: The name of the RNN model. An initial name is provided
            which indicates the time this class was instantiated.
        units: The number of LSTM units to be created on both LSTM encoder
            and decoder.
        encoder: A `model.Encoder` layer that comprises the encoding phase of
            the Seq2Seq model.
        decoder: A `model.Decoder` layer that comprises the decoding phase of
            the Seq2Seq model.
        concat_output: A concatenation layer to append current and future
            values. Usable only when training the model.
    """

    def __init__(
        self,
        input_length: int,
        horizon: int,
        n_features: int,
        model_name: Optional[str] = None,
        units: Optional[int] = 125,
        **kwargs
    ):
        if model_name is None:
            now = datetime.now()
            model_name = 'BitcoinRNN-' + now.isoformat()

        super().__init__(name=model_name, **kwargs)

        self.input_length = input_length
        self.horizon = horizon
        self.n_features = n_features
        self.model_name = model_name
        self.units = units

        self.encoder = Encoder(
            input_length=self.input_length,
            n_features=self.n_features,
            units=self.units,
        )
        self.decoder = Decoder(
            units=self.units,
            horizon=self.horizon,
        )
        self.concat_output = tf.keras.layers.Concatenate(axis=1)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            name='Optimizer'
        )
        self.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.MeanSquaredError(name='MSE'),
            metrics=tf.keras.metrics.RootMeanSquaredError(name='RMSE'),
        )

    def call(self, inputs, training=None):
        """A method to call the model that acts as the __call__() method.

        See https://www.tensorflow.org/api_docs/python/tf/keras/Model

        Args:
            inputs: A 3D tensor input of format
                [batch_number, input_length, k_components]
            training: set whether to include current timesteps in the
                prediction.

        Returns:
            A 2D tensor of shape (input_length+horizon, 1) if `training=True`,
                otherwise a 2D tensor of shape (horizon, 1).
        """
        init_predictions = self.encoder(inputs, training=training)
        future_predictions = self.decoder(init_predictions, training=training)

        if training:
            return self.concat_output([init_predictions, future_predictions])

        return future_predictions
