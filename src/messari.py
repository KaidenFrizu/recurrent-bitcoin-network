import requests

from .exceptions import MessariException

from typing import Union
from typing import Optional

import pandas as pd


class MessariObject:
    def __init__(self, response: requests.models.Response):
        if response.status_code != 200:
            exp = MessariException(response)
            exp.raise_for_status()

        self.response = response


class Metrics(MessariObject):
    def __init__(self, response: requests.models.Response):
        super().__init__(response=response)

        self.timestamp = self.response['status']['timestamp']
        self.elapsed = self.response['status']['elapsed']
        self.data = pd.DataFrame(self.response['data']['metrics'])
        self.data.set_index('metric_id', inplace=True)
        self.data.sort_index(inplace=True)

    def query(self, metric_id: str) -> pd.Series:
        return self.data.loc[metric_id]

    def get_free_metrics(self, return_df: bool = False
                        ) -> Union[pd.DataFrame, list[str]]:

        select = self.data['role_restriction'].isna()

        if return_df:
            return self.data[select]

        return self.data[select].index.to_list()

    def get_bitcoin_metrics(self, return_df: bool = False
                           ) -> Union[pd.DataFrame, list[str]]:

        res = self.get_free_metrics(return_df=return_df)

        # These metrics are either not supported by Bitcoin metrics or it
        # contains missing values in Messari API
        metrics_to_remove = [
            'cg.sply.circ', 'sply.liquid', 'txn.tfr.erc20.cnt',
            'txn.tfr.erc721.cnt', 'reddit.subscribers',
            'reddit.active.users'
        ]

        if return_df:
            return res.drop(metrics_to_remove)

        return [item for item in res if item not in metrics_to_remove]


class Timeseries(MessariObject):
    def __init__(self, response: requests.models.Response):
        super().__init__(response=response)

        self.timestamp = self.response['status']['timestamp']
        self.elapsed = self.response['status']['elapsed']

        self.parameters = self.response['data']['parameters']
        self.schema = self.response['data']['schema']
        self.data = pd.DataFrame(self.response['data']['values'],
                                 columns = self.parameters['columns'])

    def get_structured_data(self) -> pd.DataFrame:
        df = self.data.copy()

        df['timestamp'] = df['timestamp'].str[:10]
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d')

        df['metric'] = self.schema['metric_id']

        df = df.melt(id_vars=['timestamp','metric'], var_name='submetric')

        return df
