import numpy as np


def create_ts_batch(X, y, length, horizon, window_rate=1):
    x_result = []
    y_result = []

    start = 0

    while start+length+horizon <= len(y):
        x_result.append(X[start:start+length].tolist())
        y_result.append(y[start+length:start+length+horizon].tolist())

        start += window_rate

    x_result = np.array(x_result, dtype='float64')
    y_result = np.array(y_result, dtype='float64').reshape((-1,horizon,1))

    return x_result, y_result
