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
import matplotlib.pyplot as plt


class PlotPrediction:
    """Here"""

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
        """Here"""
        xindex = pd.date_range(start=date, periods=self.input_length)
        priceindex = pd.date_range(
            start=date,
            periods=self.input_length+self.horizon,
            )

        return self.features.loc[xindex], self.targets.loc[priceindex]

    def plot_predict(
        self,
        ypred,
        ytest,
        return_initial: Optional[bool] = True,
        return_legend: Optional[bool] = True,
        plot_title: Optional[str] = None,
        return_ax_only: Optional[bool] = False,
        **kwargs
    ) -> Union[tuple[plt.figure, plt.axes], plt.axes]:
        """Here"""
        fig, ax = plt.subplots(figsize=(12, 5))

        ypred_index = ytest.index[-(self.horizon+1):]
        ypred_insert = ytest[-(self.horizon+1)]

        ypred = ypred.numpy().reshape(self.horizon)
        ypred = np.insert(ypred, 0, ypred_insert)
        ypred = pd.Series(ypred, index=ypred_index)

        ax = ypred.plot(**kwargs)

        if return_initial:
            ax = ytest.plot(color='black', label='actual')

        if return_legend:
            ax.legend()

        ax.set_title(plot_title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')

        if return_ax_only:
            return ax

        return fig, ax


def plot_timeseries_data(
    df: pd.DataFrame,
    plot_title: Optional[str] = None,
    title_adjust_y: Optional[int] = None,
    savefile: Optional[str] = None,
) -> tuple[plt.figure, plt.axes]:
    """Here"""
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
