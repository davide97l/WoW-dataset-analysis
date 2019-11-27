import pandas as pd
import os
import numpy as np

pd.set_option('display.max_columns', 500)

input_dir = 'warcraft-avatar-history/'
output_dir = 'warcraft-avatar-history/'


def main():
    wow = pd.read_csv(input_dir + 'wowa_data.csv')

    # automatically annotate some labels
    labels = []
    for index, row in wow.iterrows():
        # check if killer
        # high level players, like pvp matches
        if row['lvl_end'] >= 60 and row['n_pvp'] >= 40 and row['n_maps'] <= 50:
            labels.append("killer")
        # check if socializer
        # low level players, like enjoying the game and socialize
        elif row['lvl_end'] <= 15 and row['lvl_speed'] <= 1 and row['n_city'] >= 10 and row['evolution'] <= 10:
            labels.append("socializer")
        # check if achiever
        # like growing his avatar
        elif row['time_hours'] >= 100 and row['evolution'] >= 25 and row['lvl_speed'] >= 8:
            labels.append("achiever")
        # check if explorer
        # like exploring the world
        elif row['time_hours'] >= 150 and row['n_maps'] >= 50 and row['lvl_speed'] <= 4\
                and row['evolution'] <= 25 and row['n_pvp'] <= 10:
            labels.append("explorer")
        else:
            labels.append(None)

    # add the new column to the dataset
    labels = np.asarray(labels)
    wow["behaviour"] = pd.Series(labels)

    print("Unique players: {}".format(wow.shape[0]))
    print("Killers: {}".format((labels == "killer").sum()))
    print("Socializers: {}".format((labels == "socializer").sum()))
    print("Achievers: {}".format((labels == "achiever").sum()))
    print("Explorers: {}".format((labels == "explorer").sum()))

    print("Average level speed: {}".format(wow["lvl_speed"].mean()))
    print("Average level evolution: {}".format(wow["evolution"].mean()))
    print("Average time (hours): {}".format(wow["time_hours"].mean()))
    print("Average maps visited: {}".format(wow["n_maps"].mean()))

    print("Average time on beginner maps: {}".format(wow["map_beginner"].mean()))
    print("Average time on low maps: {}".format(wow["map_low"].mean()))
    print("Average time on medium maps: {}".format(wow["map_medium"].mean()))
    print("Average time on advanced maps: {}".format(wow["map_advanced"].mean()))
    print("Average time on pvp maps: {}".format(wow["n_pvp"].mean()))
    print("Average time on city maps: {}".format(wow["n_city"].mean()))

    wow.set_index('ID', inplace=True)

    print(wow.head())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pd.DataFrame(wow).to_csv(output_dir + 'wowa_data_semi_labeled.csv', index=True)


if __name__ == "__main__":
    main()
