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
from typing import Optional
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer


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
        self.scaler = MinMaxScaler()
        self.normalizer = PowerTransformer()

    def apply_svd(
        self,
        xtrain: pd.DataFrame,
        xtest: pd.DataFrame,
        k_components: Optional[int] = None,
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

        if self.k_components != k_components:
            self.k_components = k_components
            self.svdtransformer = TruncatedSVD(self.k_components)

        xdata = pd.DataFrame(
            self.svdtransformer.fit_transform(xdata.values),
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

        init_scale = self.scaler.fit_transform(xdata.values)
        final_scale = self.normalizer.fit_transform(init_scale)
        xdata = pd.DataFrame(
            final_scale,
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
            A tuple of batch datasets (x_batched, y_batched).
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