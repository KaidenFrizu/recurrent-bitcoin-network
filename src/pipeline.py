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
import model
import transformer
import plotting


class ModelPipeline:
    """A general class used as a guideline for a model pipeline given
    a set of parameters described from an `.ini` file (model.ini).

    Args:
        ini_dir: A directory to the `.ini` config file (model.ini)
        section: The section of the config file to be chosen. Defaults to
            `DEFAULT` config args provided.

    Attributes:
        input_length
        n_features
        horizon
        k_components
        model_name
        units
        window_rate
        tensorboard_dir
        checkpoint_dir
        transformer
        tfmodel
        tbcallback
        checkpoint
    """

    def __init__(
        self,
        input_length: int,
        n_features: int,
        horizon: int,
        model_name: str,
        encoder_units: int,
        decoder_units: int,
        window_rate: int,
        use_svd: Optional[bool] = False,
        k_components: Optional[int] = None,
        is_functional: Optional[bool] = True,
    ):
        self.input_length = input_length
        self.n_features = n_features
        self.horizon = horizon
        self.k_components = k_components
        self.model_name = model_name
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.window_rate = window_rate
        self.use_svd = use_svd
        self.is_functional = is_functional
        self.plotfunc = None

        self.transformer = transformer.DataTransformer(
            input_length=self.input_length,
            horizon=self.horizon,
            window_rate=self.window_rate,
            k_components=self.k_components,
        )

        self._setup_model()
        self._set_callback_dir()
        self._set_callbacks()

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

    def _set_callbacks(self):
        self.tbcallback = tf.keras.callbacks.TensorBoard(
            log_dir=self.tensorboard_dir,
            histogram_freq=1,
        )
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_dir,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
        )
        self.nancallback = tf.keras.callbacks.TerminateOnNaN()
        self.scheduler = tf.keras.callbacks.LearningRateScheduler(
            lambda ep, rate: rate if ep <= 100 else rate * tf.math.exp(-0.01)
        )
        self.callback_list = [
            self.tbcallback,
            self.checkpoint,
            self.nancallback,
            self.scheduler,
        ]

    def _setup_model(self):
        if self.use_svd:
            n_features_to_use = self.k_components
        else:
            n_features_to_use = self.n_features

        if self.is_functional:
            self.tfmodel = model.functional_rnn(
                input_length=self.input_length,
                horizon=self.horizon,
                n_features=n_features_to_use,
                model_name=self.model_name,
                encoder_units=self.encoder_units,
                decoder_units=self.decoder_units
            )
        else:
            self.tfmodel = model.BitcoinRNN(
                input_length=self.input_length,
                horizon=self.horizon,
                n_features=n_features_to_use,
                model_name=self.model_name,
                encoder_units=self.encoder_units,
                decoder_units=self.decoder_units
            )

    def _set_plotfunction(self, features: pd.DataFrame, targets: pd.Series):
        self.plotfunc = plotting.PlotPrediction(
            features=features,
            targets=targets,
            input_length=self.input_length,
            horizon=self.horizon,
        )

    def plot_predict(self, date: str, **kwargs):
        xpred, yplot = self.plotfunc._select_data(date)
        ypred = self.predict(xpred)

        return self.plotfunc.plot_predict(ypred, date, **kwargs)


    def transform_data(
        self,
        xtrain: pd.DataFrame,
        xtest: Optional[pd.DataFrame] = None,
    ):
        """Here"""
        if xtest is not None:
            xtrain, xtest = self.transformer.normalize(xtrain, xtest)
        else:
            xtrain = self.transformer.normalize(xtrain)

        if self.use_svd:
            if xtest is not None:
                xtrain, xtest = self.transformer.apply_svd(xtrain, xtest)
            else:
                xtrain = self.transformer.apply_svd(xtrain)

        if xtest is not None:
            return xtrain, xtest

        return xtrain

    def model_train(
        self,
        xtrain: pd.DataFrame,
        xtest: pd.Series,
        ytrain: pd.DataFrame,
        ytest: pd.Series,
        epochs: int,
        batch_size: Optional[int] = 128,
        new_callback_dir: Optional[bool] = True,
        **kwargs
    ):
        """A method that is passed to the Tensorflow model `.fit()` method
        with initial parameters.

        Both `features` and `targets` are converted to numpy arrays to
        support the required data type for Tensorflow model inputs, provided
        that both `features` and `targets` are a list of `pd.DataFrame` and
        `pd.Series` respectively.

        Args:
            xtrain: Features in the shape of `(batch_size,
                n_timesteps, n_features)` to be used for model training.
            ytrain: Targets in the shape of `(batch_size, n_timesteps,
                1)` to be used as a guideline for prediction given `features`.
            batch_size: The number of batches used per epoch.
            epochs: The number of epochs
            new_callback_dir: Sets whether to set another TensorBoard
                directory for another training run.

        Returns:
            A history of loss per epoch.
        """
        self._set_plotfunction(
            features=pd.concat([xtrain, xtest]),
            targets=pd.concat([ytrain, ytest]),
        )

        xtrain, xtest = self.transform_data(xtrain, xtest)

        xtrain, ytrain = self.transformer.create_dataset(xtrain, ytrain)
        xtest, ytest = self.transformer.create_dataset(xtest, ytest)

        xtrain = np.array([df.to_numpy() for df in xtrain])
        ytrain = np.array([tgt.to_numpy() for tgt in ytrain])
        xtest = np.array([df.to_numpy() for df in xtest])
        ytest = np.array([tgt.to_numpy() for tgt in ytest])

        if new_callback_dir:
            self._set_callback_dir()
            self._set_callbacks()

        hist = self.tfmodel.fit(
            x=xtrain,
            y=ytrain,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(xtest, ytest),
            callbacks=self.callback_list,
            **kwargs
        )

        return transformer.HistoryTransformer(hist=hist, name=self.model_name)

    def predict(self, x: pd.DataFrame, **kwargs):
        """Here"""
        x = self.transform_data(x)
        x = np.array([x.values])

        return self.tfmodel(x, **kwargs)

    def reload(self, checkpoint_path: Optional[str] = None, **kwargs):
        """Reload current model weights.

        Args:
            checkpoint_path: The checkpoint path containing model weights.
                defaults to `logs/models/{model_name}`.
            kwargs: Addition key-value pairs to be passed on `.load_weights()`
                method
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir

        self.tfmodel.load_weights(checkpoint_path, **kwargs)


def load_pipeline(filepath: str, section: str):
    """Loads a ModelPipeline object from an ini file.

    Args:
        ini_dir: A directory to the `.ini` config file (model.ini)
        section: The section of the config file to be chosen. Defaults to
            `DEFAULT` config args provided.

    Returns:
        A `ModelPipeline` object.
    """
    parser = configparser.ConfigParser()
    parser.read(filepath)
    parser_args = {}

    for key, value in parser.items(section=section):
        if key in ('is_functional', 'use_svd'):
            parser_args[key] = parser.getboolean(section=section, option=key)
            continue

        if value == '':
            parser_args[key] = None
            continue

        try:
            parser_args[key] = int(value)
        except ValueError:
            parser_args[key] = str(value)

    return ModelPipeline(**parser_args)


if __name__ == '__main__':
    load_pipeline('config/model.ini', 'BASEMODEL')
