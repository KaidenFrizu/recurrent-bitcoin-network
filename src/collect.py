import messari

from typing import Optional
from typing import Literal

from requests.sessions import Session

from _url import MESSARI_METRICS_URL
from _url import MESSARI_TS_URL

class Collector(object):

    def __init__(self, headers: Optional[dict] = None):
        if headers is None:
            headers = dict()

        self.headers = headers
        self.sess = Session()
        self.sess.headers.update({'content-type':'application/json'})
        self.sess.headers.update(self.headers)

    def restart_session(self):
        self.sess = Session()
        self.sess.headers.update({'content-type':'application/json'})
        self.sess.headers.update(self.headers)

    def get_metrics(self) -> messari.Metrics:
        url = MESSARI_METRICS_URL

        response = self.sess.get(url)
        response.raise_for_status()

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
            columns = list()

        url = MESSARI_TS_URL.format(assetkey, metric_id)
        params = {
        'start':start, 'end':end, 'interval':interval,
        'columns':','.join(columns), 'order':order,
        'format':formatting, 'timestamp-format':timestamp_format
        }

        response = self.sess.get(url, params=params)
        response.raise_for_status()

        return messari.Timeseries(response)
