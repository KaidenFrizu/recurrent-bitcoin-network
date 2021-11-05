import requests

from _url import MESSARI_ASSETS_URL
from _url import MESSARI_METRICS_URL
from _url import MESSARI_PROFILE_URL
from _url import MESSARI_TS_URL


def get_assets(sess=None, fields=['id','slug','symbol'],
               page=None, sort=None, limit=20, **kwargs):

    if sess is None:
        sess = requests.session()

    params = {
        'fields':','.join(fields),
        'page':page, 'sort':sort,
        'limit':limit
    }

    return sess.get(MESSARI_ASSETS_URL, params=params, **kwargs)


def get_profile(assetkey, fields=None, as_markdown=False,
                sess=None, **kwargs):

    if sess is None:
        sess = requests.session()
    
    if fields is None:
        fields = list()

    url = MESSARI_PROFILE_URL.format(assetkey)
    params = {
        'fields':','.join(fields),
        'as-markdown':as_markdown
    }

    return sess.get(url, params=params, **kwargs)

def get_metrics(assetkey, fields=None, sess=None, **kwargs):

    if sess is None:
        sess = requests.session()

    url = MESSARI_METRICS_URL.format(assetkey)
    params ={'fields':fields}

    return sess.get(url, params=params, **kwargs)


def get_asset_timeseries(assetkey, metric_id,
                         start='2016-01-01', end='2020-12-31',
                         interval='1d', timestamp_format='rfc3339',
                         columns=None, order=None, formatting='json',
                         sess=None, **kwargs):

    if sess is None:
        sess = requests.session()

    url = MESSARI_TS_URL.format(assetkey, metric_id)
    params = {
        'start':start, 'end':end, 'interval':interval,
        'columns':columns, 'order':order, 'format':formatting,
        'timestamp-format':timestamp_format
    }

    return sess.get(url, params=params, **kwargs)

