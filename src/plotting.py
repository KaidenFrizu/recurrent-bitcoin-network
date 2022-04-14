"""Contains the plotting functions for displaying model predictions and
training results.

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
from typing import Union
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


class PlotPrediction:
    """A class used for plot model predictions.

    Although it is possible to use this class explicitly, it is recommended
    to use `pipeline.ModelPipeline` instead which this class is treated as an
    attribute.

    Args:
        features: A data frame of features to be used for prediction.
        targets: A series of values to be predicted.
        input_length: The number of steps to be used as inputs at a time.
        horizon: The number of timesteps for forecasting.

    Attributes:
        features
        targets
        input_length
        horizon
    """

    def __init__(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        input_length: int,
        horizon: int,
    ):
        self.features = features
        self.targets = targets
        self.input_length = input_length
        self.horizon = horizon

    def select_data(self, date: str):
        """Selects a series of data points from a given starting date.

        Note that the starting date indicates the start date of feature inputs
        where it is calculated by `input_length` and `horizon` to return a
        particular series of features and target values.

        Args:
            date: The start date of the features to be selected.

        Returns:
            A tuple of selected features and its corresponding targets.
        """
        xindex = pd.date_range(start=date, periods=self.input_length)
        priceindex = pd.date_range(
            start=date,
            periods=self.input_length+self.horizon,
            )

        return self.features.loc[xindex], self.targets.loc[priceindex]

    def plot_predict(
        self,
        ypred: tf.Tensor,
        ytest: pd.Series,
        return_initial: Optional[bool] = True,
        return_legend: Optional[bool] = True,
        plot_title: Optional[str] = None,
        return_preds_only: Optional[bool] = False,
        **kwargs
    ) -> Union[tuple[plt.Figure, plt.Axes], plt.Axes]:
        """Plots the given data points through a line plot.

        This method is particularly unstable as it is yet to account for
        positions and plot scaling. This is best to be used as is.

        Args:
            ypred: A series of predicted values.
            ytest: A series of values to be tested from predictions.
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
        ypred_index = ytest.index[-(self.horizon+1):]
        ypred_insert = ytest[-(self.horizon+1)]

        ypred = ypred.numpy().reshape(self.horizon)
        ypred = ypred.cumsum()
        ypred = ypred + ypred_insert
        ypred = np.insert(ypred, 0, ypred_insert)
        ypred = pd.Series(ypred, index=ypred_index)

        if return_preds_only:
            return ypred.plot(**kwargs)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax = ypred.plot(**kwargs)

        if return_initial:
            ax = ytest.plot(color='black', label='actual')

        if return_legend:
            ax.legend()

        ax.set_title(plot_title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')

        return fig, ax


def plot_timeseries_data(
    df: pd.DataFrame,
    plot_title: Optional[str] = None,
    title_adjust_y: Optional[float] = None,
    savefile: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots the time series features row-wise.

    Args:
        df: A dataframe of time series data points in wide format.
        plot_title: The title of the plot to be added.
        title_adjust_y: The y-component of the title position.
        savefile: The name of the file for plot saving.

    Returns:
        A tuple of `pyplot.Figure` and `pyplot.Axes`.
    """
    if plot_title is None:
        plot_title = 'Time-series of Blockchain-derived Data'
    if title_adjust_y is None:
        title_adjust_y = 0.91

    n_features = df.columns.unique('metric').shape[0]
    fig, ax = plt.subplots(nrows=n_features, ncols=1, sharex=True)

    for i, colname in zip(np.arange(n_features), df.columns.unique('metric')):
        feature_data = df[colname]
        feature_data.plot(
            figsize=(12, 2.75*n_features),
            title=colname,
            ax=ax[i],
            lw=1.5,
        )
        ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.suptitle(plot_title, y=title_adjust_y)

    if savefile is not None:
        fig.savefig(savefile, bbox_inches='tight', dpi=300)

    return fig, ax


if __name__ == '__main__':
    # Script is used to debug in this .py file

    data = pd.read_csv('raw/train.csv', parse_dates=['timestamp'])
    data = data.pivot_table(
        index='timestamp',
        columns=['metric', 'submetric'],
        values='value'
    )
    plot_timeseries_data(data)
