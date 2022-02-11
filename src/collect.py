import messari
import exceptions

from typing import Optional
from typing import Literal
from typing import Union

import requests

from _url import MESSARI_METRICS_URL
from _url import MESSARI_TS_URL


class APICollector:

    def __init__(
        self,
        headers: Optional[dict] = None,
        params: Optional[dict] = None
    ):

        if headers is None:
            headers = {}
        if params is None:
            params = {}

        self.headers = headers
        self.params = params
        self.sess = requests.Session()
        self.sess.headers.update(self.headers)
        self.sess.params.update(self.params)

    def restart_session(self):
        self.sess = requests.Session()
        self.sess.headers.update(self.headers)
        self.sess.params.update(self.params)

    def get(self, url: Union[str, bytes], **kwargs) -> requests.models.Response:
        return self.sess.get(url, **kwargs)


class MessariCollector(APICollector):

    def __init__(
        self,
        headers: Optional[dict] = None,
        params: Optional[dict] = None
    ):

        super().__init__(headers = headers, params=params)

    def get_metrics(self) -> messari.Metrics:
        url = MESSARI_METRICS_URL

        response = self.get(url)
        response.raise_for_status()

        if response.status_code == 200:
            return messari.Metrics(response)

    def get_timeseries(
        self,
        assetkey: str,
        metric_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: Literal['1m','5m','15m','30m','1h','1d','1w'] = '1d',
        timestamp_format: \
            Literal['unix-millisecond','unix-second','rfc3339'] = 'rfc3339',
        columns: Optional[list[str]] = None,
        order: Optional[Literal['asc', 'desc']] = None,
        formatting: Literal['csv', 'json'] = 'json',
    ) -> messari.Timeseries:

        # Instantiate a dummy list for passing to .join() method
        if columns is None:
            columns = []

        url = MESSARI_TS_URL.format(assetkey, metric_id)
        params = {
        'start':start, 'end':end, 'interval':interval,
        'columns':','.join(columns), 'order':order,
        'format':formatting, 'timestamp-format':timestamp_format
        }

        response = self.get(url, params=params)

        if response.status_code == 200:
            return messari.Timeseries(response)

        return exceptions.MessariException(response)


class NasdaqCollector(APICollector):

    def __init__(
        self,
        headers: Optional[dict] = None,
        params: Optional[dict] = None
    ):

        super().__init__(headers=headers, params=params)

    # To do list
    def get_timeseries(self):
        pass
