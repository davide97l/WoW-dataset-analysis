import pandas as pd
import utils
import os
from datetime import datetime
import sys

sys.path.append('..')

pd.set_option('display.max_columns', 500)

input_dir = '../warcraft-avatar-history/'
output_dir = '../warcraft-avatar-history/'
player_dir = 'players/'


def main():
    wowah = pd.read_csv(input_dir + 'wowah_data.csv', parse_dates=True, keep_date_col=True)
    if not os.path.exists(player_dir):
        os.makedirs(player_dir)

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

    # replace timestamp with datetime object
    wowah['timestamp'] = wowah['timestamp'].apply(lambda x: datetime.strptime(x, '%m/%d/%y %H:%M:%S').isocalendar()[1])\
        .astype('int32')

    # create new dataset
    wow = wowah.drop_duplicates(['timestamp', 'ID']).sort_values(['timestamp'], ascending=True)\
        .sort_values(['timestamp'], ascending=True)
    wow.drop(['zone'], axis=1, inplace=True)
    wow.drop(['race'], axis=1, inplace=True)
    wow.drop(['class'], axis=1, inplace=True)
    wow.drop(['guild'], axis=1, inplace=True)
    wow.drop(['level'], axis=1, inplace=True)
    wow.set_index(['ID', 'timestamp'], drop=False, inplace=True)

    print("Unique players: {}".format(wow.shape[0]))

    wow['lvl_start'] = wowah.groupby(['ID', 'timestamp'])['level'].min()
    wow['lvl_end'] = wowah.groupby(['ID', 'timestamp'])['level'].max()
    # avatar's level increase during the week
    wow['evolution'] = wow['lvl_end'] - wow['lvl_start']
    # avatar's average level
    wow['lvl_avg'] = (wow['lvl_end'] + wow['lvl_start']) / 2
    wow.drop(['lvl_start'], axis=1, inplace=True)
    wow.drop(['lvl_end'], axis=1, inplace=True)

    # weekly hours played
    wow['time_hours'] = (wowah.groupby(['ID', 'timestamp'])['ID'].count() / 6).round(3).apply(utils.smooth_hours)

    ids = wow['ID'].tolist()
    max_id = max(ids)
    c = 4
    n_weeks = 52

    print(wow.head())

    for idx in ids:
        print("processing ID: {} of {}".format(idx, max_id))
        temp_wow = wow.loc[wow['ID'] == idx]
        temp_wow.drop(['ID'], axis=1, inplace=True)
        temp_wow.set_index(['timestamp'], drop=True, inplace=True)
        temp_wow['status'] = 0
        # status 0 = ING (in-game)
        # status 1 = ATC (about to churn)
        # status 0 = CHR (churned)
        # status 0 = the player hasn't begun playing yet
        weeks = temp_wow.index.tolist()
        index = 1
        last_lvl_avg = 0
        while index <= n_weeks:
            # add rows relative to weeks not present in the dataset (inactivity weeks)
            if index not in weeks:
                temp_wow.loc[index] = {'evolution': 0, 'lvl_avg': last_lvl_avg,
                                       'time_hours': 0, 'status': 0}
            else:
                last_lvl_avg = temp_wow.at[index, 'lvl_avg']
            index += 1
        index = 1
        active_weeks = 0
        temp_wow['current_absence'] = 0  # weeks since last played (first week is always 0)
        temp_wow['week_present_ratio'] = 0.0  # active weeks divided by the number of weeks since first game
        # set player activity status (0 = active, 1 = about to churn, 2 = churned
        while index <= n_weeks:
            if index == 0:
                while temp_wow.at[index, 'time_hours'] == 0:
                    temp_wow.at[index, 'status'] = -1  # hasn't begun playing yet
                    index += 1
                    active_weeks += 1
            if temp_wow.at[index, 'time_hours'] == 0:  # if no played during week index
                if index > 1:
                    if temp_wow.at[index - 1, 'time_hours'] > 0:
                        temp_wow.at[index, 'current_absence'] = 1
                    else:
                        temp_wow.at[index, 'current_absence'] = temp_wow.at[index-1, 'current_absence'] + 1
                count = 1
                # checks if the player hasn't played for c weeks in a row
                for i in range(1, c):  # 1 to c-1
                    if (index + i) < temp_wow.shape[0] and temp_wow.at[index + i, 'time_hours'] == 0:
                        count += 1
                if count == c:  # if the players hasn't played for c weeks in a row
                    # set about to churn weeks
                    for i in range(1, c+1):  # 1 to c
                        if (index - i) > 0 and temp_wow.at[index - i, 'status'] == 0:
                            temp_wow.at[index - i, 'status'] = 1  # about to churn
                    # set churn weeks
                    while index <= temp_wow.shape[0] and temp_wow.at[index, 'time_hours'] == 0:
                        temp_wow.at[index, 'status'] = 2  # churned
                        if index > 1:
                            temp_wow.at[index, 'current_absence'] = temp_wow.at[index - 1, 'current_absence'] + 1
                        temp_wow.at[index, 'week_present_ratio'] = active_weeks / index
                        index += 1
                else:
                    index += 1
            else:
                if index > 1:
                    if temp_wow.at[index-1, 'time_hours'] > 0:
                        temp_wow.at[index, 'current_absence'] = 0
                    else:
                        temp_wow.at[index, 'current_absence'] = temp_wow.at[index-1, 'current_absence']
                active_weeks += 1
                index += 1
            temp_wow.at[index-1, 'week_present_ratio'] = active_weeks / (index-1)

        temp_wow['week_present_ratio'] = temp_wow['week_present_ratio'].round(3)
        temp_wow.sort_values(['timestamp'], ascending=True, inplace=True)
        pd.DataFrame(temp_wow).to_csv(player_dir + 'player_' + str(idx) + '.csv', index=True)


if __name__ == "__main__":
    main()
