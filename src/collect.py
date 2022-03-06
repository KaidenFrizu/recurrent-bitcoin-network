"""A collection of Python classes for retrieving Bitcoin-related features
through web API connection.

The usual class export is `MessariCollector` which facilitates GET requests to
Messari API. It is subclassed under a generic `APICollector` framework for
Python requests module.

Example:
    collector = MessariCollector()
    metrics = collector.get_metrics()
    bitcoin_data = collector.get_timeseries(assetkey='BTC', metric_id='price')

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
from typing import Literal
from typing import Union
import re
import time
import requests
from tqdm import tqdm
import apiobjects
from _url import MESSARI_METRICS_URL
from _url import MESSARI_TS_URL


# pylint: disable=too-many-arguments

class APICollector:
    """A base web API collector to be used as a tool for data collection.

    Requests through `APICollector` uses `requests.Session` to perform a
    variety of requests. This is mostly used for GET requests.

    It is not advisable to use this class unless for a specific purpose. It is
    recommended to use classes that are a subclass of `APICollector`.

    Args:
        headers: A dictionary of headers to be passed for every request made.
            This would be stored as an attribute with the same name and is
            passed to `sess` attribute that is a `requests.Session` object.

    Attributes:
        headers: A dictionary of headers to be passed for every request made.
            This is passed to `sess` attribute that is a `requests.Session`
            object.
        sess: A `requests.Session` object used for performing web API
            requests.
    """

    def __init__(self, headers: Optional[dict] = None):
        if headers is None:
            headers = {}

        self.headers = headers
        self.sess = requests.Session()

        self.sess.headers.update(self.headers)

    def restart_session(self, update_headers: bool = True):
        """Restarts the current session by overwriting the attribute `sess`
        with a new `requests.Session` object.

        Args:
            update_headers: A boolean whether to use the existing headers to
                the new `requests.Session` object.
        """
        self.sess = requests.Session()

        if update_headers:
            self.sess.headers.update(self.headers)

    def get_request(
        self,
        url: Union[str, bytes],
        raise_status: bool = False,
        **kwargs
    ) -> requests.Response:
        """Performs a GET request to the provided url through the current
        session `sess` with the provided headers and parameters.

        Args:
            url: A url to be used for a GET request.
            raise_status: Determine whether to raise exceptions when
                returned a non-200 status code response.
            **kwargs: Key-value pairs of optional arguments to be passed on
                `sess` attribute. See `requests.request()`_ for the list of
                possible arguments.

        Returns:
            A `requests.Response` object

        Raises:
            ConnectionError: The request made has lost its connection.
            HTTPError: The request made did not return a `Response<200>`
                status code.

        .. _requests.request():
            https://docs.python-requests.org/en/latest/api/#requests.request
        """
        response = self.sess.get(url, **kwargs)
        if raise_status:
            response.raise_for_status()

        return response


class MessariCollector(APICollector):
    """A class dedicated to collect data from Messari.io through web API
    requests.

    This class is intended for web API requests to Messari.io for details
    regarding the API, see the `Messari API documetation`_.

    Args:
        api_key: A string of API key to be included in the headers
        headers: A dictionary of headers to be passed for every request made.
            This is passed to `sess` attribute that is a `requests.Session`
            object.

    Attributes:
        headers: A dictionary of headers to be passed for every request made.
            This is passed to `sess` attribute that is a `requests.Session`
            object.
        sess: A `requests.Session` object used for performing web API
            requests.

    Example:
        collector = MessariCollector()
        metrics = collector.get_metrics()
        bitcoin_data = collector.get_timeseries(
            assetkey='BTC',
            metric_id='price'
        )

    .. _Messari API documentation:
        https://messari.io/api/docs
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        headers: Optional[dict] = None
    ):
        if headers is None:
            headers = {}

        headers['x-messari-api-key'] = api_key

        super().__init__(headers=headers)

    def get_metrics(self) -> apiobjects.MessariMetrics:
        """Retrieves a response request containing possible metrics from a
        cryptocurrency asset and stored using the `messari.Metrics` object.

        The given metrics have a corresponding `metric_id` that is useful for
        retrieving time series data using the `get_timeseries()` method. This
        is applicable to all available cryptocurrency assets in Messari web
        API.

        Returns:
            An `apiobjects.MessariMetrics` object.

        Raises:
            ConnectionError: The request made has lost its connection.
            HTTPError: The request made did not return a `Response<200>`
                status code.
        """
        response = self.get_request(MESSARI_METRICS_URL, raise_status=True)

        return apiobjects.MessariMetrics(response)

    def get_timeseries(
        self,
        assetkey: str,
        metric_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: Optional[str] = '1d',
        columns: Optional[list[str]] = None,
        order: Optional[Literal['asc', 'desc']] = None,
        formatting: Optional[Literal['csv', 'json']] = 'json',
        timestamp_format: Optional[str] = 'rfc3339',
    ) -> apiobjects.MessariTimeseries:
        """Retrieves a response request of time series from Messari API with
        the given parameters.

        The args provided are used as one of the parameters for a GET request.
        These args have its own default values corresponding to the default
        values given from the `Messari API documentation`_.

        For any given interval, at most 2016 data points will be returned,
        regardless on its granularity. Exceeding the maximum range will return
        a non-200 status code. However, it can be resolved by reducing the
        date range specified in `start` and `end` args.

        The data points are sorted in ascending order: The earliest date would
        be shown first in the data.

        Args:
            assetkey: An ID, name, or an abbreviation of a particular
                cryptocurrency to be retrieved.
            metric_id: An ID retrieved through `get_metrics()` method that
                denotes the metric to be retrieved.
            start: A date in %Y-%m-%d format that sets the starting point of
                the time series data to be collected. A default start date
                would be provided by Messari if not specified.
            end: A date in %Y-%m-%d format that sets the ending point of the
                time series data to be collected. A default end date would be
                provided by Messari if not specified.
            interval: Denotes the granularity of the time series data points
                by minute, hour, day, or week. One of '1m', '5m', '15m',
                '30m', '1h', '1d', or '1w'.
            columns: A list of columns to be selected from the time series
                data.
            order: Determines whether the data starts in the earliest or in
                the latest date.
            formatting: Denotes the type of content in the response object
                to be returned by Messari API.
            timestamp_format: Determines the format of the timestamp column
                in the time series data. One of 'unix-millisecond',
                'unix-second', or 'rfc3339'

        Returns:
            A `apiobjects.MessariTimeseries` object.

        Raises:
            ConnectionError: The request made has lost its connection.
            HTTPError: The request made did not return a `Response<200>`
                status code.

        .. _Messari API documentation:
        https://messari.io/api/docs#operation/Get%20Asset%20timeseries
        """
        # Instantiate a dummy list for passing to .join() method
        if columns is None:
            columns = []

        url = MESSARI_TS_URL.format(assetkey, metric_id)
        params = {
            'start': start, 'end': end, 'interval': interval,
            'columns': ','.join(columns), 'order': order,
            'format': formatting, 'timestamp-format': timestamp_format
        }

        response = self.get_request(url, raise_status=True, params=params)

        return apiobjects.MessariTimeseries(response)

    def get_fulldata(
        self,
        assetkey: str,
        start: str,
        end: str,
        only_freemetrics: Optional[bool] = True,
    ) -> list[apiobjects.MessariTimeseries, ...]:
        """Retrieves the complete time series data from each available metric.

        It uses the method `get_timeseries()` multiple times to retrieve time
        series data with different metrics. A tqdm progress bar would be
        shown during data collection process.

        Args:
            assetkey: An ID, name, or an abbreviation of a particular
                cryptocurrency to be retrieved.
            start: A date in %Y-%m-%d format that sets the starting point of
                the time series data to be collected. A default start date
                would be provided by Messari if not specified.
            end: A date in %Y-%m-%d format that sets the ending point of the
                time series data to be collected. A default end date would be
                provided by Messari if not specified.

        Returns:
            A list of `apiobjects.MessariTimeseries` with different metrics.

        Raises:
            ConnectionError: The request made has lost its connection.
            HTTPError: The request made did not return a `Response<200>`
                status code.
        """
        metrics_list = self.get_metrics()

        if only_freemetrics:
            metrics = metrics_list.get_free_metrics()
        else:
            metrics = metrics_list.data

        data_list = []

        with tqdm(total=len(metrics.index)) as pbar:
            for metric in metrics.index:
                statuscode = 0

                while statuscode != 200:
                    try:
                        result = self.get_timeseries(
                            assetkey=assetkey, metric_id=metric,
                            start=start, end=end
                        )
                        statuscode = result.response.status_code
                        data_list.append(result)
                        pbar.update(1)

                    except requests.exceptions.HTTPError as err:
                        errdata = err.response.json()
                        statuscode = err.response.status_code
                        errmessage = errdata['status']['error_message']
                        errtxt = f'Response [{statuscode}]: {errmessage}'
                        pbar.write(errtxt)

                        if statuscode == 429:
                            timer_int = re.findall(r"\d+", errmessage)
                            cooldown = int(timer_int[0]) + 1
                            time.sleep(cooldown)
                        elif statuscode >= 500:
                            pbar.update(1)
                            break
                        else:
                            raise err

        return data_list
