"""Contains a collection of API objects to be used for data preprocessing.
These would take a response from a GET request and transforms it into class
object.

These are used alongside with `collect.py` to retrieve data through various
web APIs. These would serve as a building block for creating datasets and
preprocessing for Tensorflow model inputs. It is not recommened to import
the following classes unless used with `collect.py` unless for a very specific
reason.

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
from datetime import datetime
import requests
import pandas as pd


class APIObject:
    """A core class for creating API objects for data preprocessing. It
    contains general verification for successful requests upon class creation.

    Args:
        response: A `requests.Response` object to be stored as an attribute
            of the same name.

    Attributes:
        response: The `requests.Response` object passed

    Raises:
            HTTPError: The request made did not return a `Response<200>`
                status code.
    """

    def __init__(self, response: requests.Response):
        response.raise_for_status()
        self.response = response


class MessariBaseObject(APIObject):
    """A generic class for creating classes from responses in Messari web API.
    Details on the request status including timestamp of GET request and
    request duration are also processed through this class.

    Args:
        response: A `requests.Response` object to be stored as an attribute
            of the same name.

    Attributes:
        content: The content returned in json (dict) format
        timestamp: A string of a timestamp that signifies the time the reuqest
            took place.
        elapsed: The duration of the GET request from its request to its
            returned response.

    Raises:
            HTTPError: The request made did not return a `Response<200>`
                status code.
    """

    def __init__(self, response: requests.Response):
        super().__init__(response=response)
        self.content = self.response.json()
        self.timestamp = datetime.strptime(
            self.content['status']['timestamp'][:19],
            '%Y-%m-%dT%H:%M:%S'
        )
        self.elapsed = self.content['status']['elapsed'] * 0.01

    def __repr__(self):
        value_template = '<Messari API Response {0} on {1} in {2} seconds>'
        print_value = value_template.format(
            str(self.response.status_code),
            self.timestamp.strftime('%b %d, %Y'),
            str(round(self.elapsed, 2))
        )

        return print_value


class MessariMetrics(MessariBaseObject):
    """An API object used to retrieve the possible metrics to be used for
    requesting time series data in Messari.

    Args:
        response: A `requests.Response` object to be stored as an attribute
            of the same name.

    Attributes:
        content: The content returned in json (dict) format
        timestamp: A string of a timestamp that signifies the time the reuqest
            took place.
        elapsed: The duration of the GET request from its request to its
            returned response.
        data: A `pandas.DataFrame` that shows a list of metrics with
            corresponding descriptions and `metric_id` for requesting time
            series data in Messari web API.
    """

    def __init__(self, response: requests.Response):
        super().__init__(response=response)
        self.data = pd.DataFrame(self.content['data']['metrics'])
        self.data.set_index('metric_id', inplace=True)
        self.data.sort_index(inplace=True)

    def get_free_metrics(self) -> pd.DataFrame:
        """Returns a filtered table of metrics that is publicly accessible
        through web API request in Messari. This is determined under
        `role_restriction` column.

        There are some metrics that are only available when applied through
        a subscription. These metrics would be omitted as such.

        Returns:
            A `pandas.DataFrame` of available (non-premium) metrics.
        """
        mask = self.data['role_restriction'].isna()

        return self.data[mask]

    def get_bitcoin_metrics(self) -> pd.DataFrame:
        """Retrieves all possible metrics for BTC (Bitcoin) time series.
        Besides its filter for free metrics, certain metrics would be removed
        because of incomplete values or they do not support BTC.

        Returns:
            A `pandas.DataFrame` of available BTC metrics.
        """
        free_data = self.get_free_metrics()

        metrics_to_remove = [
            'cg.sply.circ', 'sply.liquid', 'txn.tfr.erc20.cnt',
            'txn.tfr.erc721.cnt', 'reddit.subscribers',
            'reddit.active.users'
        ]

        return free_data.drop(metrics_to_remove)


class MessariTimeseries(MessariBaseObject):
    """A class dedicated to store time series data retrieved through Messari
    web API.

    The data would be preprocessed into a clean and machine-readable format
    especially for Tensorflow models.

    Args:
        response: A `requests.Response` object to be stored as an attribute
            of the same name.

    Attributes:
        asset_id: A unique id of a given cryptocurrency asset.
        asset_key: The abbreviation for a given cryptocurrency asset.
        asset_name: The name of the cryptocurrency asset.
        end_date: A `datetime` object which denotes the end date of the given
            data received.
        format_: The format of the data returned (csv or json).
        interval: The granularity of the data point returned.
        metric_description: A description of the given metric returned.
        metric_id: A unique id of a given metric
        metric_name: The name of the metric
        order: Shows whether the data is in ascending (asc) or descending
            (desc) order
        parameters: A dictionary of API parameters returned containing details
            regarding the given request.
        schema: A dictionary of additional status information regarding
            Messari web API request.
        start_date: A `datetime` object which denotes the start date of the
            given data received.
        slug: An alternative identifier for a given cryptocurrency asset.
        submetrics: A dictionary of column descriptions returned in the
            GET request.
        timestamp_format: Denotes the format of the returned timestamp.
    """

    def __init__(self, response: requests.models.Response):
        super().__init__(response=response)
        self.parameters = self.content['data']['parameters']
        self.schema = self.content['data']['schema']
        self.asset_id = self.parameters['asset_id']
        self.asset_key = self.parameters['asset_key']
        self.asset_name = self.schema['name']
        self.start_date = datetime.strptime(
            self.parameters['start'][:19],
            '%Y-%m-%dT%H:%M:%S'
        )
        self.end_date = datetime.strptime(
            self.parameters['end'][:19],
            '%Y-%m-%dT%H:%M:%S'
        )
        self.interval = self.parameters['interval']
        self.order = self.parameters['order']
        self.format_ = self.parameters['format']
        self.timestamp_format = self.parameters['timestamp_format']
        self.metric_id = self.schema['metric_id']
        self.metric_name = self.schema['name']
        self.metric_description = self.schema['description']
        self.submetrics = self.schema['values_schema']

        self._load_data()

    def _load_data(self):
        """A private method used to load the data through `pandas.DataFrame`.

        General preprocessing is done through this method which includes
        conversion of timestamps to a `datetime` object, and inclusion of
        column names.
        """
        self.data = pd.DataFrame(
            self.content['data']['values'],
            columns=self.parameters['columns'],
        )

        if self.timestamp_format == 'rfc3339':
            self.data['timestamp'] = self.data['timestamp'].str[:19]
            self.data['timestamp'] = pd.to_datetime(
                self.data['timestamp'],
                format='%Y-%m-%dT%H:%M:%S',
            )

    def get_melt_data(
        self,
        id_vars: Optional[Union[str, list[str]]] = None,
        value_vars: Optional[Union[str, list[str]]] = None,
        var_name: Optional[str] = 'submetric',
        **kwargs
    ) -> pd.DataFrame:
        """Returns a melted `pandas.DataFrame` data for export or dataset
        joining using pandas `melt()` method.

        By default, `timestamp` and `metric` columns would be used as
        `id_vars` while the rest of the columns would be treated as
        `value_vars` in melting the time series data.

        For more info, see `pandas.DataFrame.melt()`_

        Args:
            id_vars: Column(s) that would be treated as identifier variables.
            value_vars: Columns(s) that would be used to unpivot.
            var_name: The name that would be used for the variable column.
            **kwargs: Additional key-value pairs to be passed in
                `pandas.DataFrame.melt()`.

        Returns:
            A `pandas.DataFrame` of melted data

        .. _pandas.DataFrame.melt():
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.melt.html#pandas.DataFrame.melt
        """
        if id_vars is None:
            id_vars = ['timestamp', 'metric']

        dataframe = self.data.copy()
        dataframe['metric'] = self.schema['metric_id']
        result_df = dataframe.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=var_name,
            **kwargs
        )

        return result_df
