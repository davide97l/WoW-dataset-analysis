import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', 500)

input_dir = 'warcraft-avatar-history/'
output_dir = 'warcraft-avatar-history/'


def main():
    wow = pd.read_csv(input_dir + 'wowa_data_semi_labeled.csv')

    classes = ['killer', 'socializer', 'achiever', 'explorer']

    # encode behaviours type with numerical values
    labels = []
    for index, row in wow.iterrows():
        if row['behaviour'] in classes:
            labels.append(classes.index(row['behaviour']))
        else:
            labels.append(-1)
    labels = np.asarray(labels)
    wow["behaviour"] = pd.Series(labels)

    temp_wow = wow.copy()

    # one-hot encode string features
    wow = pd.concat([wow, pd.get_dummies(wow['race'], prefix='race')], axis=1)
    wow = pd.concat([wow, pd.get_dummies(wow['class'], prefix='class')], axis=1)
    wow.drop(['race'], axis=1, inplace=True)
    wow.drop(['class'], axis=1, inplace=True)

    # prepare training and test set
    wow_train = wow[wow['behaviour'] != -1]
    train_labels = np.array(wow_train['behaviour'])
    wow_test = wow[wow['behaviour'] == -1]
    wow_train.drop(['behaviour'], axis=1, inplace=True)
    wow_test.drop(['behaviour'], axis=1, inplace=True)
    train = np.array(wow_train)
    test = np.array(wow_test)

    # train model and make predictions
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train, train_labels)
    pred = model.predict(test)

    # update unknown labels
    i = 0
    for index, row in temp_wow.iterrows():
        if row['behaviour'] == -1:
            temp_wow.at[index, 'behaviour'] = pred[i]
            i += 1

    # convert back behaviours type as strings
    labels = []
    for index, row in temp_wow.iterrows():
        labels.append(classes[row['behaviour']])
    labels = np.asarray(labels)
    temp_wow["behaviour"] = pd.Series(labels)

    print("Killers: {}".format((labels == "killer").sum()))
    print("Socializers: {}".format((labels == "socializer").sum()))
    print("Achievers: {}".format((labels == "achiever").sum()))
    print("Explorers: {}".format((labels == "explorer").sum()))

    wow.set_index('ID', inplace=True)

    print(temp_wow.head())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pd.DataFrame(temp_wow).to_csv(output_dir + 'wowa_data_labeled.csv', index=True)


if __name__ == "__main__":
    main()
