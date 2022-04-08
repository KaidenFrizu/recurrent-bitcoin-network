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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PlotPrediction:
    """Here"""

    def __init__(self, input_length: int, horizon: int):
        self.input_length = input_length
        self.horizon = horizon


def plot_timeseries_data(
    df: pd.DataFrame,
    plot_title: Optional[str] = None,
    savefile: Optional[str] = None,
) -> tuple[plt.figure, plt.axes]:
    """Here"""
    if plot_title is None:
        plot_title = 'Time-series of Blockchain-derived Data'

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

    fig.suptitle(plot_title, y=0.91)

    if savefile is not None:
        fig.savefig(savefile, bbox_inches='tight', dpi=300)

    return fig, ax


if __name__ == '__main__':
    df = pd.read_csv('raw/train.csv', parse_dates=['timestamp'])
    df = df.pivot_table(
        index='timestamp',
        columns=['metric', 'submetric'],
        values='value'
    )
    plot_timeseries_data(df)
