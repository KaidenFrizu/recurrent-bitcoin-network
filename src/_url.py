"""A compilation of URLs to be used for GET requests.

This is used in particular to classes dedicated for data collection.

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
MESSARI_BASE_URL = 'https://data.messari.io/api/v1'
MESSARI_BASE_URL_V2 = 'https://data.messari.io/api/v2'
MESSARI_ASSETS_URL = MESSARI_BASE_URL_V2 + '/assets'
MESSARI_METRICS_URL = MESSARI_BASE_URL + '/assets/metrics'
MESSARI_TS_URL = MESSARI_BASE_URL + '/assets/{0}/metrics/{1}/time-series'
