import pandas as pd
import os
from datetime import datetime, timedelta

pd.set_option('display.max_columns', 500)

input_dir = '../warcraft-avatar-history/'
output_dir = '../warcraft-avatar-history/'


def main():
    wowah = pd.read_csv(input_dir + 'wowah_data_small.csv', parse_dates=True, keep_date_col=True)

    # clean dataset

    # rename the dataset to make column names more clear
    wowah.rename({'char': 'ID',
                  ' level': 'level',
                  ' race': 'race',
                  ' charclass': 'class',
                  ' zone': 'zone',
                  ' guild': 'guild',
                  ' timestamp': 'timestamp'}, axis=1, inplace=True)
    # drop rows with wrong timestamp format
    wowah.tmp_len = wowah.timestamp.str.len()
    wowah.drop(wowah[wowah.tmp_len != 17].index, inplace=True)
    wowah = wowah.sort_values(['ID'], ascending=True)

    print("Unique sessions: {}".format(wowah.shape[0]))

    # returns 1 if a player hasn't played for the next 30 days after the measured timestamp
    def check_unsubscribed(x):
        ID = x['ID']
        ts = x['timestamp']
        a = wowah[wowah['ID'] == ID]
        b = a[a['timestamp'] < ts + timedelta(days=30)]
        if b.count()['ID'] > 0:
            return 0
        return 1

    # replace timestamp with datetime object
    wowah['timestamp'] = wowah['timestamp'].apply(lambda x: datetime.strptime(x, '%d/%m/%y %H:%M:%S'))
    wowah['unsubscribed'] = wowah[['timestamp', 'ID']].apply(check_unsubscribed, axis=1)

    print(wowah.head())

    """if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pd.DataFrame(wow).to_csv(output_dir + 'wowa_data.csv', index=True)"""


if __name__ == "__main__":
    main()
