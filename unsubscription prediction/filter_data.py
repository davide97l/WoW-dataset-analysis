import pandas as pd
import os

pd.set_option('display.max_columns', 500)

input_dir = '../warcraft-avatar-history/'
output_dir = '../warcraft-avatar-history/'


def main():
    wowah = pd.read_csv(input_dir + 'wowah_data.csv', parse_dates=True, keep_date_col=True)

    max_ID = 10000
    ID_range = [i for i in range(1, max_ID+1)]

    wowah = wowah[wowah['char'].isin(ID_range)]

    print(wowah.head())

    print("Unique sessions: {}".format(wowah.shape[0]))
    print("Unique players: {}".format(len(wowah['char'].unique())))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pd.DataFrame(wowah).to_csv(output_dir + 'wowah_data_filtered_' + str(max_ID) + '.csv', index=False)


if __name__ == "__main__":
    main()
