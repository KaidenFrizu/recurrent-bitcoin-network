import pandas as pd

def get_table(requests_data):
    content = requests_data.json()['data']

    cols = content['parameters']['columns'][1:]
    newcols = [
        ' - '.join([content['schema']['metric_id'], item])
        for item in cols
    ]
    sample_df = pd.DataFrame(
        content['values'],
        columns = ['timestamp'] + newcols
    )
    sample_df['timestamp'] = pd.to_datetime(
        sample_df['timestamp'],
        format='%Y-%m-%dT%H:%M:%S'
    )

    return sample_df

