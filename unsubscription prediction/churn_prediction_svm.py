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
history = 7
only_last_abt_week = False  # if True consider only the last ABT week and set the others as ING


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
            # consider only the last ABT week
            if only_last_abt_week and index < n_weeks and ph.at[index, 'status'] == 1 and ph.at[index + 1, 'status'] == 1:
                ph.at[index, 'status'] = 0
            if ph.at[index, 'status'] != 2 and index > history-1:  # not churned and has an history
                row = []
                for j in range(history):
                    ev = ph.at[index-j, 'evolution']
                    la = ph.at[index-j, 'lvl_avg']
                    th = ph.at[index-j, 'time_hours']
                    ca = ph.at[index-j, 'current_absence']
                    pr = ph.at[index-j, 'week_present_ratio']
                    row += [ev, la, th, ca, pr]
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
    history = 7
    accuracy on test set: 0.797
    --- ING labels ---
    precision: 0.944
    recall: 0.805
    fscore: 0.869
    --- ATC labels ---
    precision: 0.433
    recall: 0.756
    fscore: 0.550
    
    (considering churn weeks only the ast week of the churning period)
        accuracy on test set: 0.895
    --- ING labels ---
    precision: 0.976
    recall: 0.908
    fscore: 0.941
    --- ATC labels ---
    precision: 0.381
    recall: 0.719
    fscore: 0.498
    """


if __name__ == "__main__":
    main()
