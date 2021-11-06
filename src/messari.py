import requests
import json
import time

import pandas as pd
from tqdm import tqdm

from _url import MESSARI_ASSETS_URL
from _url import MESSARI_METRICS_URL
from _url import MESSARI_PROFILE_URL
from _url import MESSARI_TS_URL

from formatting import get_table


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


def download(filename=None, to_csv_kwargs=None, **kwargs):

    with open('_metrics.json', 'r') as f:
        metrics = json.load(f)
        df_metrics = pd.read_json(metrics)

    data = list()
    pbar = tqdm(df_metrics['metric_id'])

    for metric in pbar:
        pbar.set_description(
            'Now retrieving: {0}'.format(metric)
        )

        success = False
        while not success:
            res = get_asset_timeseries(
                assetkey='BTC',
                metric_id=metric,
                **kwargs
            )

            if res.status_code == 200:
                data.append(get_table(res))
                success = True
                time.sleep(1)

            elif res.status_code // 100 == 5:
                pbar.write(
                    'Request [{0}] in {1}'.format(
                        res.status_code, metric
                    )
                )
                time.sleep(5)
                break

            else:
                pbar.set_description(
                    '[Request {0}] | Now retrieving: {1}'.format(
                        res.status_code, metric
                    )
                )
                time.sleep(10)

    data = pd.concat(data, axis=1).sort_index()

    if filename is not None:
        data.to_csv(filename, **to_csv_kwargs)

    return data


def load(filename, overwrite=False, read_csv_kwargs=dict(),
         to_csv_kwargs=dict(), **kwargs):

    if not overwrite:
        try:
            return pd.read_csv(filename, **read_csv_kwargs)
        except FileNotFoundError:
            return download(filename, to_csv_kwargs, **kwargs)

    return download(filename, to_csv_kwargs, **kwargs)

