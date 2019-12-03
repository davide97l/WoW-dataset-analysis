import pandas as pd
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import utils
import sys
sys.path.append('..')

pd.set_option('display.max_columns', 500)

input_dir = '../warcraft-avatar-history/'
output_dir = '../warcraft-avatar-history/'
player_dir = 'players/'
history = 10
only_last_abt_week = False  # if True consider only the last ABT week and set the others as ING


def main():
    wowah = pd.read_csv(input_dir + 'wowa_data_labeled.csv')
    id_list = np.array(wowah[wowah['behaviour'] == 'killer']['ID'])
    csv_files = glob.glob(player_dir + "/*.csv")
    abt = []
    ing = []
    n_files = len(csv_files)
    for i, filename in enumerate(csv_files):
        name = int(filename.split('.')[0].split('\\')[1].split('_')[1])
        #if name not in id_list:
        #    continue
        print("elaborating file {} of {}".format(i + 1, n_files))
        ph = pd.read_csv(filename)  # get player history
        ph["timestamp"] = pd.to_numeric(ph["timestamp"], downcast='integer')
        ph.set_index(['timestamp'], inplace=True)
        n_weeks = 52
        for index in range(1, n_weeks+1):
            # consider only the last ABT week
            if only_last_abt_week and index < n_weeks and ph.at[index, 'status'] == 1 and ph.at[index+1, 'status'] == 1:
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
    ing_labels = np.array([0 for i in range(len(ing))])[:len(abt)]

    abt = np.array(abt)
    ing = np.array(ing)[:len(abt), :]

    X, y = shuffle(np.concatenate((abt, ing)), np.concatenate((abt_labels, ing_labels)))

    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # prepare training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # reshape input to be 3D [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test), verbose=2,
                    shuffle=False)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    y_pred = model.predict(X_test, verbose=0)
    y_pred = y_pred > 0.5

    acc = accuracy_score(y_pred, y_test)
    print("accuracy on test set: {:.3f}".format(acc))
    utils.print_metrics(y_pred, y_test)

    """
    history = 10
    accuracy on test set: 0.783
    --- ING labels ---
    precision: 0.855
    recall: 0.749
    fscore: 0.798
    --- ATC labels ---
    precision: 0.710
    recall: 0.829
    fscore: 0.765
    
    (considering churn weeks only the ast week of the churning period)
    accuracy on test set: 0.850
    --- ING labels ---
    precision: 0.880
    recall: 0.828
    fscore: 0.853
    --- ATC labels ---
    precision: 0.821
    recall: 0.875
    fscore: 0.847
    """


if __name__ == "__main__":
    main()
