"""Contains the custom data transormation for RNN model datasets and plotting.

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
from typing import Union
from typing import Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler


class DataTransformer:
    """A custom class dedicated for transforming datasets into RNN inputs and
    batch datasets.

    Args:
        input_length: The number of data points over time to be fed as input.
        horizon: The number of future timesteps to cast a prediction.
        window_rate: The interval between batches on its beginning timestep.
        k_components: The number of components used in SVD.

    Attributes:
        input_length
        k_components
        horizon
        window_rate
        svdtransformer
        scaler
        normalizer
    """

    def __init__(
        self,
        input_length: Optional[int] = None,
        horizon: Optional[int] = None,
        window_rate: Optional[int] = None,
        k_components: Optional[int] = None,
    ):
        if k_components is None:
            k_components = 2

        self.input_length = input_length
        self.k_components = k_components
        self.horizon = horizon
        self.window_rate = window_rate

        self.svdtransformer = TruncatedSVD(self.k_components)
        self.scaler = MinMaxScaler(feature_range=(-5, 5))

    def apply_svd(
        self,
        xtrain: pd.DataFrame,
        xtest: Optional[pd.DataFrame] = None,
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
        """Applies SVD transformation to x inputs.

        Args:
            xtrain: A `pandas.DataFrame` of features in the training set.
            xtest: A `pandas.DataFrame` of features in the test set.

        Returns:
            A tuple of SVD transformed values: (xtrain, xtest).
        """
        if xtest is not None:
            date_slice = xtest.index[0]
            xdata = pd.concat([xtrain, xtest])
        else:
            xdata = xtrain

        xdata = pd.DataFrame(
            self.svdtransformer.fit_transform(xdata.values),
            index=xdata.index
        )

        if xtest is not None:
            return xdata[:date_slice], xdata[date_slice:]

        return xdata

    def normalize(
        self,
        xtrain: pd.DataFrame,
        xtest: Optional[pd.DataFrame] = None,
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
        """Applies scaling to normalize features.

        Args:
            xtrain: A `pandas.DataFrame` of features in the training set.
            xtest: A `pandas.DataFrame` of features in the test set.

        Returns:
            A tuple of normalized values: (xtrain, xtest).
        """
        if xtest is not None:
            date_slice = xtest.index[0]
            xdata = pd.concat([xtrain, xtest])
        else:
            xdata = xtrain

        init_scale = self.scaler.fit_transform(xdata.values)
        xdata = pd.DataFrame(
            init_scale,
            index=xdata.index,
            columns=xdata.columns,
        )

        if xtest is not None:
            return xdata[:date_slice], xdata[date_slice:]

        return xdata

    def transform(
        self,
        xtrain: pd.DataFrame,
        xtest: Optional[pd.DataFrame] = None,
        use_diff: Optional[bool] = False,
        use_svd: Optional[bool] = False,
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
        if use_diff:
            if xtest is not None:
                xtest = xtest.diff(axis=0).fillna(0)

            xtrain = xtrain.diff(axis=0).fillna(0)

        if xtest is not None:
            xtrain, xtest = self.normalize(xtrain, xtest)
        else:
            xtrain = self.normalize(xtrain)

        if use_svd:
            if xtest is not None:
                xtrain, xtest = self.apply_svd(xtrain, xtest)
            else:
                xtrain = self.apply_svd(xtrain)

        if xtest is not None:
            return xtrain, xtest

        return xtrain

    def create_dataset(
        self,
        xdata: pd.DataFrame,
        ydata: pd.Series,
        return_train_y: Optional[bool] = False,
        to_cumsum: Optional[bool] = False,
        use_diff: Optional[bool] = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Creates time series batch datasets based on the given window rate.

        Note that the creation of samples are given in a constant window rate
        which does not involve random sampling.

        Args:
            xdata: A `pandas.DataFrame` of features.
            ydata: A `pandas.Series` of target values.
            return_train_y: Determines whether to include current timesteps
                of y values in the batches.
            to_cumsum: Determines whether to output target values as a
                cumulative sum that is set before the first target value.

        Returns:
            A tuple of batch datasets in `np.ndarray` (x_batched, y_batched).
        """
        xresult = []
        yresult = []

        if use_diff:
            ydata = ydata.diff().fillna(0)

        start = 0
        while start + self.input_length + self.horizon <= ydata.shape[0]:
            stop_window = start + self.input_length

            xres = xdata[start:stop_window]
            xresult.append(xres.to_numpy())

            if return_train_y:
                res = ydata[start:stop_window+self.horizon]
            elif to_cumsum:
                res = ydata[stop_window-1:stop_window+self.horizon]
                res = res - res[0]
                res = res[1:]
            else:
                res = ydata[stop_window:stop_window+self.horizon]

            yresult.append(res.to_numpy())
            start += self.window_rate

        return np.array(xresult), np.array(yresult)


class HistoryTransformer:
    """A class dedicated to preprocess the model history data for summaries.

    This is used for preprocessing model performance through given metrics for
    plotting and visualizations.

    Args:
        hist: A history from a model callback. This is taken during model
            training phase.
        name: The name of the transforer, usually indicates the SVD type used.
        metric: The metric used for determining performance.

    Attributes:
        hist
        name
        metric
    """

    def __init__(
        self,
        hist: tf.keras.callbacks.History,
        name: str,
        metric: Optional[str] = 'RMSE',
    ):
        self.hist = hist
        self.name = name
        self.metric = metric

    def show_results(self):
        """Preprocess the given history from callbacks for plotting and
        visualization purposes.

        Returns:
            A `pd.DataFrame` of metric values per epoch in long format.
        """
        data_dict = {
            'train': self.hist.history[self.metric],
            'test': self.hist.history['val_'+self.metric]
        }

        result = pd.DataFrame(data_dict)
        result['epoch'] = result.index
        result['svd_type'] = self.name

        return result.melt(
            id_vars=['epoch', 'svd_type'],
            var_name='dataset',
            value_name=self.metric,
        )
