"""Contains the custom data pipeline for RNN model predition.

The pipeline configuration is based on a config .ini file that contains the
parameters to be used for the model.

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
import os
import configparser
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import model


class ModelPipeline:
    """A general class used as a guideline for a model pipeline given
    a set of parameters described from an `.ini` file (model.ini).

    Args:
        ini_dir: A directory to the `.ini` config file (model.ini)
        section: The section of the config file to be chosen. Defaults to
            `DEFAULT` config args provided.

    Attributes:
        input_length
        horizon
        k_components
        model_name
        units
        window_rate
        tensorboard_dir
        checkpoint_dir
        svd
        scaler
        tfmodel
        tbcallback
        checkpoint
    """

    def __init__(
        self,
        ini_dir: str,
        section: Optional[str] = 'DEFAULT',
    ):
        parser = configparser.ConfigParser()
        parser.read(ini_dir)
        parser_section = parser[section]

        self.input_length = int(parser_section['input_length'])
        self.horizon = int(parser_section['horizon'])
        self.k_components = int(parser_section['k_components'])
        self.model_name = parser_section['model_name']
        self.units = int(parser_section['units'])
        self.window_rate = int(parser_section['window_rate'])

        if self.k_components > 0:
            self.svd = TruncatedSVD(
                n_components=self.k_components,
                algorithm='arpack'
            )

        self.scaler = MinMaxScaler()

        self.tfmodel = model.BitcoinRNN(
            input_length=self.input_length,
            horizon=self.horizon,
            n_features=self.k_components,
            model_name=self.model_name,
            units=self.units
        )

        self._set_callback_dir()
        self.tbcallback = tf.keras.callbacks.TensorBoard(
            log_dir=self.tensorboard_dir,
            histogram_freq=1,
        )
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_dir,
            monitor='loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
        )

    def _set_callback_dir(self):
        tbdir = os.path.join(
            os.getcwd(),
            '..',
            'logs',
            'tensorboard',
            self.model_name,
            datetime.now().strftime('%Y-%m-%dT%H%M%S'),
        )
        ckpt = os.path.join(
            os.getcwd(),
            '..',
            'logs',
            'weights',
            self.model_name,  # Would treat as folder directory
            self.model_name,  # For filename use
        )

        self.tensorboard_dir = os.path.abspath(tbdir)
        self.checkpoint_dir = os.path.abspath(ckpt)

    def apply_svd(
        self,
        xtrain: pd.DataFrame,
        xtest: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Applies SVD transformation to x inputs.

        Args:
            xtrain: A `pandas.DataFrame` of features in the training set.
            xtest: A `pandas.DataFrame` of features in the test set.

        Returns:
            A tuple of SVD transformed values: (xtrain, xtest).
        """
        date_slice = xtest.index[0]
        xdata = pd.concat([xtrain, xtest])
        if self.k_components > 0:
            xdata = pd.DataFrame(
                self.svd.fit_transform(xdata.values),
                index=xdata.index
            )

        return xdata[:date_slice], xdata[date_slice:]

    def normalize(
        self,
        xtrain: pd.DataFrame,
        xtest: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Applies scaling to normalize features.

        Args:
            xtrain: A `pandas.DataFrame` of features in the training set.
            xtest: A `pandas.DataFrame` of features in the test set.

        Returns:
            A tuple of normalized values: (xtrain, xtest).
        """
        date_slice = xtest.index[0]
        xdata = pd.concat([xtrain, xtest])
        xdata = pd.DataFrame(
            self.scaler.fit_transform(xdata.values),
            index=xdata.index,
            columns=xdata.columns,
        )

        return xdata[:date_slice], xdata[date_slice:]

    def create_dataset(
        self,
        xdata: pd.DataFrame,
        ydata: pd.Series,
        return_train_y: Optional[bool] = False,
    ) -> tuple[list[pd.DataFrame], list[pd.Series]]:
        """Creates time series batch datasets based on the given window rate.

        Note that the creation of samples are given in a constant window rate
        which does not involve random sampling.

        Args:
            xdata: A `pandas.DataFrame` of features.
            ydata: A `pandas.Series` of target values.
            return_train_y: Determines whether to include current timesteps
                of y values in the batches.

        Returns:
            A tuple of batch datasets (x_batched, y_batched)
        """
        xresult = []
        yresult = []

        start = 0
        while start + self.input_length + self.horizon <= ydata.shape[0]:
            stop_window = start + self.input_length
            xresult.append(xdata[start:stop_window])

            if return_train_y:
                yresult.append(ydata[start:stop_window+self.horizon])
            else:
                yresult.append(ydata[stop_window:stop_window+self.horizon])

            start += self.window_rate

        return xresult, yresult

    def model_train(
        self,
        train_features: list[pd.DataFrame],
        train_targets: list[pd.Series],
        test_features: list[pd.DataFrame],
        test_targets: list[pd.Series],
        batch_size: Optional[int] = 96,
        epochs: Optional[int] = 300,
        **kwargs
    ):
        """A method that is passed to the Tensorflow model `.fit()` method
        with initial parameters.

        Both `features` and `targets` are converted to numpy arrays to
        support the required data type for Tensorflow model inputs, provided
        that both `features` and `targets` are a list of `pd.DataFrame` and
        `pd.Series` respectively.

        Args:
            train_features: Features in the shape of `(batch_size,
                n_timesteps, n_features)` to be used for model training.
            train_targets: Targets in the shape of `(batch_size, n_timesteps,
                1)` to be used as a guideline for prediction given `features`.
            batch_size: The number of batches used per epoch.
            epochs: The number of epochs
        """
        train_features = np.array([df.to_numpy() for df in train_features])
        train_targets = np.array([tgt.to_numpy() for tgt in train_targets])
        test_features = np.array([df.to_numpy() for df in test_features])
        test_targets = np.array([tgt.to_numpy() for tgt in test_targets])

        self.tfmodel.fit(
            x=train_features,
            y=train_targets,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(test_features, test_targets),
            callbacks=[self.tbcallback, self.checkpoint],
            **kwargs
        )

    def reload(self, checkpoint_path: Optional[str] = None, **kwargs):
        """Reload current model weights

        Args:
            checkpoint_path: The checkpoint path containing model weights.
                defaults to `logs/models/{model_name}`.
            kwargs: Addition key-value pairs to be passed on `.load_weights()`
                method
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir

        self.tfmodel.load_weights(checkpoint_path, **kwargs)
