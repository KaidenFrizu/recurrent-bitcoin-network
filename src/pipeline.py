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
            lambda ep, rate: rate if ep <= 150 else rate * tf.math.exp(-0.001)
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

    def plot_predict(
        self,
        date: str,
        return_initial: Optional[bool] = True,
        return_legend: Optional[bool] = True,
        plot_title: Optional[str] = None,
        return_preds_only: Optional[bool] = False,
        **kwargs
    ):
        """Shows the plot prediction based on a given date.

        The data used for plotting are from the data passed during training,
        `.model_train()`. The data are merged together to have a continuous
        prediction on part of train and test data.

        Args:
            date: A date that indicates the start date used for retrieving
                predictors.
            return_initial: Determines whether to plot the initial and actual
                values in the given time period.
            return_legend: Shows whether or not to return a plot legend.
            plot_title: The name of the plot to be shown above.
            return_preds_only: Shows whether to return only `pyplot.Axes`
                only. This is used for parent-level plotting or requires
                additional plot arguments.

        Returns:
            A tuple of `pyplot.Figure` and `pyplot.Axes` depending on the
                value of `return_ax_only`.
        """
        xtest, ytest = self.plotfunc.select_data(date)
        ypred = self.predict(xtest, transform=False)

        return self.plotfunc.plot_predict(
            ypred=ypred,
            ytest=ytest,
            return_initial=return_initial,
            return_legend=return_legend,
            plot_title=plot_title,
            return_preds_only=return_preds_only,
            **kwargs
        )

    def transform_data(
        self,
        xtrain: pd.DataFrame,
        xtest: Optional[pd.DataFrame] = None,
        use_diff: Optional[bool] = False,
    ):
        """Transforms the data into optimized, model-readable form for
        prediction.

        If both `xtrain` and `xtest` are supplied, they are concatenated
        together before SVD is applied.

        Args:
            xtrain: Data containing the features for training.
            xtest: Data containing the features for testing.
            use_diff: Sets whether to apply first differencing to the given
                data.
            use_svd: Sets whether to apply SVD.

        Returns:
            Two `pd.DataFrame`s if `xtest` is also supplied, else it returns
                one `pd.DataFrame`.
        """
        return self.transformer.transform(
            xtrain=xtrain,
            xtest=xtest,
            use_diff=use_diff,
            use_svd=self.use_svd,
        )

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
        xtrain, xtest = self.transform_data(xtrain, xtest, use_diff=True)

        self._set_plotfunction(
            features=pd.concat([xtrain, xtest]),
            targets=pd.concat([ytrain, ytest]),
        )

        xtrain, ytrain = self.transformer.create_dataset(
            xdata=xtrain,
            ydata=ytrain,
            use_diff=True
        )
        xtest, ytest = self.transformer.create_dataset(
            xdata=xtest,
            ydata=ytest,
            use_diff=True
        )

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

    def predict(
        self,
        x: pd.DataFrame,
        transform: Optional[bool] = True,
        **kwargs
    ):
        """Predicts the future values given a series of features.

        Args:
            x: A pd.DataFrame of features used for prediction.
            transform: Decides whether or not to transform the given features
                before feeding to the model. This is generally disabled when
                the given features are already preprocessed.

        Returns:
            A `tf.Tensor` of model predictions.
        """
        if transform:
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
