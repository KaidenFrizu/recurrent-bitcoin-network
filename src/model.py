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


class Encoder(tf.keras.Model):
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
        units: int,
    ):

        super().__init__(name='Encoder')

        self.core_lstm = tf.keras.layers.LSTM(
            units=units,
            activation=tf.keras.activations.swish,
            dropout=0,
            kernel_initializer=tf.keras.initializers.RandomUniform(seed=306),
            kernel_regularizer='l2',
            bias_initializer=tf.keras.initializers.RandomUniform(seed=306),
            bias_regularizer='l1',
            name='encoder_lstm',
            return_sequences=True,
        )
        self.bilstm = tf.keras.layers.Bidirectional(
            layer=self.core_lstm,
            merge_mode='concat',
        )

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
        return self.bilstm(x, training=training)


class Decoder(tf.keras.Model):
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

        self.lstm = tf.keras.layers.LSTM(
            units=units,
            activation=tf.keras.activations.swish,
            dropout=0,
            kernel_regularizer='l1',
            kernel_initializer=tf.keras.initializers.RandomUniform(seed=306),
            bias_regularizer=None,
            bias_initializer=tf.keras.initializers.RandomUniform(seed=306),
            name='decoder_lstm',
            return_sequences=True,
        )
        self.flatlayer = tf.keras.layers.Flatten(name='flat_layer')
        self.resolve = tf.keras.layers.Dense(
            units=self.horizon,
            name='predict'
        )

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
        x = self.lstm(x, training=training)
        x = self.flatlayer(x)

        return self.resolve(x)


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
        encoder_units: The number of LSTM units to be created on LSTM encoder.
        decoder_units: The number of LSTM units to be created on LSTM encoder.
        **kwargs: Additional arguments passed on `tensorflow.keras.Model`.

    Attributes:
        input_length: The number of data points over time to be fed as input.
        horizon: The number of data points from the end of input data point
            to cast a prediction from the model.
        n_features: The number of features to be fed as past inputs.
        model_name: The name of the RNN model. An initial name is provided
            which indicates the time this class was instantiated.
        encoder_units: The number of LSTM units to be created on LSTM encoder.
        decoder_units: The number of LSTM units to be created on LSTM encoder.
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
        model_name: Optional[str] = None,
        n_features: Optional[int] = None,
        encoder_units: Optional[int] = 100,
        decoder_units: Optional[int] = 20,
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
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units

        self.input_layer = tf.keras.layers.InputLayer(
            input_shape=(self.input_length, self.n_features),
            name='input_encoder',
        )
        self.encoder = Encoder(
            units=self.encoder_units,
        )
        self.decoder = Decoder(
            units=self.decoder_units,
            horizon=self.horizon,
        )

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            name='Optimizer'
        )
        self.model_loss = tf.keras.losses.MeanSquaredError(name='MSE')
        self.model_metrics = tf.keras.metrics.RootMeanSquaredError(name='RMSE')

        self.compile(
            optimizer=self.optimizer,
            loss=self.model_loss,
            metrics=self.model_metrics,
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
        inputs = self.input_layer(inputs)
        init_predictions = self.encoder(inputs, training=training)

        return self.decoder(init_predictions, training=training)


def functional_rnn(
    input_length: int,
    horizon: int,
    model_name: Optional[str] = None,
    n_features: Optional[int] = None,
    encoder_units: Optional[int] = 100,
    decoder_units: Optional[int] = 20,
    show_summary: Optional[bool] = True,
    **kwargs
):
    """Creates the RNN model through Tensorflow Functional API architecture.

    This returns the same BitcoinRNN model but built through TensorFlow's
    Functional API. This is generally used for features that are not present
    in Subclassing API such as model graphs.

    Args:
        input_length:
        horizon:
        model_name:
        n_features:
        encoder_units:
        decoder_units:
        **kwargs: Key-value pair arguments to be passed in `BitcoinRNN` class.

    Returns:
        A `tensorflow.keras.Model`.
    """
    modeltemplate = BitcoinRNN(
        input_length=input_length,
        horizon=horizon,
        model_name=model_name,
        n_features=n_features,
        encoder_units=encoder_units,
        decoder_units=decoder_units,
        **kwargs
    )

    input_encoder = tf.keras.Input(
        shape=(input_length, n_features),
        name='input_encoder',
    )
    output_encoder = modeltemplate.encoder.bilstm(input_encoder)
    encoder = tf.keras.Model(
        inputs=input_encoder,
        outputs=output_encoder,
        name='Encoder'
    )

    if modeltemplate.encoder.bilstm.merge_mode == 'concat':
        encoder_units = encoder_units * 2

    input_decoder = tf.keras.Input(
        shape=(input_length, encoder_units),
        name='input_decoder',
    )
    x = modeltemplate.decoder.lstm(input_decoder)
    x = modeltemplate.decoder.flatlayer(x)
    output_decoder = modeltemplate.decoder.resolve(x)
    decoder = tf.keras.Model(
        inputs=input_decoder,
        outputs=output_decoder,
        name='Decoder'
    )

    ensemble_input = tf.keras.Input(
        shape=(input_length, n_features),
        name='Input',
    )
    encoded_vals = encoder(ensemble_input)
    decoded_vals = decoder(encoded_vals)

    rnn_model = tf.keras.Model(
        inputs=ensemble_input,
        outputs=decoded_vals,
        name=model_name
    )
    rnn_model.compile(
        optimizer=modeltemplate.optimizer,
        loss=modeltemplate.model_loss,
        metrics=modeltemplate.model_metrics,
    )

    if show_summary:
        encoder.summary()
        decoder.summary()
        rnn_model.summary()

    return rnn_model
