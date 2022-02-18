import numpy as np
from typing import Optional


def create_ts_batch(
    X, y, X_future,
    length: int,
    horizon: int,
    window_rate: Optional[int] = 1,
):
    """Create a time series batch of features and target variables to be fed
    in TensorFlow model

    Args:
        X - An array-like structure of numeric features.
        y - An array of numeric target variable with same length as X
        X_future - An array-like structure of deterministic features (dates)
        length - The number of samples taken in X for each batch
        horizon - The number of samples taken in y for each batch
        window_rate - The number between its successive batch marked within the
        time series sequence.

    Returns:
        A tuple of X and y batches in numpy.array and plotdata from y
    """
    x_result = []
    xfuture_result = []
    y_result = []
    plotdata = []

    # Iterating to create windows
    start = 0
    while start+length+horizon <= len(y):
        stop_window = start + length
        x_result.append(X[start:stop_window].tolist())
        xfuture_result.append(X_future[stop_window:stop_window+horizon].tolist())
        y_result.append(y[stop_window:stop_window+horizon].tolist())
        plotdata.append(y[start:stop_window].tolist())

        start += window_rate

    x_result = np.array(x_result, dtype='float64')
    xfuture_result = np.array(xfuture_result, dtype='float64')
    y_result = np.array(y_result, dtype='float64').reshape((-1, horizon, 1))
    plotdata_result = np.array(plotdata, dtype='float64').reshape(
        (-1, length, 1)
    )

    return (x_result, xfuture_result), y_result, plotdata_result
