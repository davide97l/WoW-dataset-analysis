import pandas as pd
import utils
import os

import seaborn as sns
sns.set()

pd.set_option('display.max_columns', 500)

input_dir = 'warcraft-avatar-history/'
output_dir = 'warcraft-avatar-history/'


def main():
    wowah = pd.read_csv(input_dir + 'wowah_data.csv', parse_dates=True, keep_date_col=True)
    zones = pd.read_csv(input_dir + 'zones.csv', encoding='iso-8859-1')

    # clean the original dataset

    # rename the dataset to make column names more clear
    wowah.rename({'char': 'ID',
                  ' level': 'level',
                  ' race': 'race',
                  ' charclass': 'class',
                  ' zone': 'zone',
                  ' guild': 'guild',
                  ' timestamp': 'timestamp'}, axis=1, inplace=True)
    # replace chinese characters with latin words
    zones['Zone_Name'].replace({'Dalaran<U+7AF6><U+6280><U+5834>': 'Dalaran Arena'}, inplace=True)
    wowah['zone'].replace({'Dalaran競技場': 'Dalaran Arena'}, inplace=True)
    # drop rows with wrong timestamp format
    wowah.tmp_len = wowah.timestamp.str.len()
    wowah.drop(wowah[wowah.tmp_len != 17].index, inplace=True)
    wowah = wowah.sort_values(['ID'], ascending=True)

    print("Unique sessions: {}".format(wowah.shape[0]))

    # create a new dataset containing unique players statistics
    wow = wowah.drop_duplicates(['ID']).sort_values(['ID'], ascending=True)
    wow.drop(['timestamp'], axis=1, inplace=True)
    wow.drop(['level'], axis=1, inplace=True)
    wow.drop(['zone'], axis=1, inplace=True)
    wow.set_index('ID', inplace=True)
    print("Unique players: {}".format(wow.shape[0]))

    # adding features to the new dataset

    wow['lvl_start'] = wowah.groupby(['ID'])['level'].min()
    wow['lvl_end'] = wowah.groupby(['ID'])['level'].max()
    # which is the final level minus the initial
    wow['evolution'] = wow['lvl_end'] - wow['lvl_start']
    # total playing time during the period analyzed
    wow['time_hours'] = wowah.groupby(['ID'])['timestamp'].count() / 6
    # relates to the evolution with time spent (lvl_start-lvl_end)/days
    wow['lvl_speed'] = wow['evolution'] / (wow['time_hours'] / 24)
    # how many guilds the player joined during the period analyzed
    wow['n_guilds'] = wowah.groupby(['ID']).apply(lambda x: x.drop_duplicates(['guild'])['guild'].count())
    # number of visited maps during the analyzed period
    wow['n_maps'] = wowah.groupby(['ID']).apply(lambda x: x.drop_duplicates(['zone'])['zone'].count())

    # filter to remove players who have played in total less than 2 hours
    wow = wow[wow['time_hours'] >= 2]

    # creates a dict ["Zone_name", "Type"] from dataset wowah
    zones_dict = zones[['Zone_Name', 'Type']].set_index(['Zone_Name']).T.to_dict('records')[0]
    wowah['zone_type'] = wowah['zone'].map(zones_dict)
    # hours spent in a zone of kind "city"
    wow['n_city'] = wowah.groupby(['ID']).apply(lambda x: (x.loc[x['zone_type'] == "City"])['zone_type'].count()) / 6
    # hours spent in a zone of kind "pvp"
    wow['n_pvp'] = wowah.groupby(['ID']).apply(lambda x: (x.loc[x['zone_type'].isin(['Arena', 'Battleground'])])
                                                          ['zone_type'].count()) / 6
    # add the min and max recommended level for each zone in the wowah dataset
    zones_min_rec = zones[zones['Type'].isin(['Zone', 'Transit', 'Sea'])][['Zone_Name', 'Min_rec_level']].set_index(['Zone_Name']).T.to_dict('records')[0]
    zones_max_rec = zones[zones['Type'].isin(['Zone', 'Transit', 'Sea'])][['Zone_Name', 'Max_rec_level']].set_index(['Zone_Name']).T.to_dict('records')[0]
    wowah['min_lvl_rec'] = wowah['zone'].map(zones_min_rec)
    wowah['max_lvl_rec'] = wowah['zone'].map(zones_max_rec)

    # hours spent in a zone of a certain difficulty
    wow['map_beginner'] = wowah.groupby(['ID']).apply(utils.count_maps_beginner) / 6
    wow['map_low'] = wowah.groupby(['ID']).apply(utils.count_maps_low) / 6
    wow['map_medium'] = wowah.groupby(['ID']).apply(utils.count_maps_medium) / 6
    wow['map_advanced'] = wowah.groupby(['ID']).apply(utils.count_maps_advanced) / 6

    print(wow.head())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pd.DataFrame(wow).to_csv(output_dir + 'wowa_data.csv', index=True)


if __name__ == "__main__":
    main()
