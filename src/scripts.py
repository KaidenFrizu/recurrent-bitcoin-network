import os
import time
import collect
from exceptions import MessariException
from tqdm import tqdm

def get_data(start, end):

    api_header = {'x-messari-api-key':os.getenv('MESSARI_API_KEY')}
    collector = collect.MessariCollector(headers=api_header)
    metrics = collector.get_metrics()

    btc_metrics = metrics.get_free_metrics()
    data_list = list()


    with tqdm(total=len(btc_metrics)) as pbar:
        pbar.set_description('Response [200]')

        for metric in btc_metrics:
            while True:
                result = collector.get_timeseries(
                    assetkey='BTC', metric_id=metric,
                    start=start, end=end
                )

                if isinstance(result, MessariException):
                    pbar.write('Response [{0}]: {1}'.format(
                        result.error_code, result.error_message
                    ))

                    if result.error_code == 429:
                        time.sleep(result.cooldown)
                        continue
                    
                    break

                data_list.append(result.get_structured_data())
                pbar.update(1)
                break

    return data_list


def get_train_data():
    print('Collecting Train Data...')
    return get_data('2016-01-01', '2020-12-31')


def get_test_data():
    print('Collecting Test Data...')
    return get_data('2021-01-01', '2021-06-30')


if __name__ == '__main__':
    import pandas as pd

    train_list = get_train_data()
    test_list = get_test_data()

    train = pd.concat(train_list, ignore_index=True)
    test = pd.concat(test_list, ignore_index=True)

    train.to_csv('../raw/train.csv', index=False)
    test.to_csv('../raw/test.csv', index=False)
