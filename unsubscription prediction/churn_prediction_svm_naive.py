import pandas as pd
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import utils
import sys
sys.path.append('..')

pd.set_option('display.max_columns', 500)

input_dir = '../warcraft-avatar-history/'
output_dir = '../warcraft-avatar-history/'
player_dir = 'players/'


def main():
    csv_files = glob.glob(player_dir + "/*.csv")
    abt = []
    ing = []
    n_files = len(csv_files)
    for i, filename in enumerate(csv_files):
        print("elaborating file {} of {}".format(i + 1, n_files))
        ph = pd.read_csv(filename)  # get player history
        ph["timestamp"] = pd.to_numeric(ph["timestamp"], downcast='integer')
        ph.set_index(['timestamp'], inplace=True)
        n_weeks = 52
        for index in range(1, n_weeks+1):
            if ph.at[index, 'status'] != 2:  # not churned
                ev = ph.at[index, 'evolution']
                la = ph.at[index, 'lvl_avg']
                th = ph.at[index, 'time_hours']
                ca = ph.at[index, 'current_absence']
                pr = ph.at[index, 'week_present_ratio']
                row = [ev, la, th, ca, pr]
                if ph.at[index, 'status'] == 0:  # active
                    ing.append(row)
                else:  # about to churn
                    abt.append(row)

    print("ABT sequences: {}".format(len(abt)))
    print("ING sequences: {}".format(len(ing)))
    abt_labels = np.array([1 for i in range(len(abt))])
    ing_labels = np.array([0 for i in range(len(ing))])

    abt = np.array(abt)
    ing = np.array(ing)[:, :len(abt)]

    X, y = shuffle(np.concatenate((abt, ing)), np.concatenate((abt_labels, ing_labels)))

    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # prepare training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # use svm
    clf = SVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    print("accuracy on test set: {:.3f}".format(acc))
    utils.print_metrics(y_pred, y_test)

    """
    accuracy on test set: 0.766
    --- ING labels ---
    precision: 0.950
    recall: 0.769
    fscore: 0.850
    --- ATC labels ---
    precision: 0.341
    recall: 0.746
    fscore: 0.468
    """


if __name__ == "__main__":
    main()
