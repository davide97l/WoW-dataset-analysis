from sklearn.metrics import precision_recall_fscore_support


def count_maps_beginner(x):
    c = 0
    x=x[['min_lvl_rec', 'max_lvl_rec']]
    for index, row in x.iterrows():
        if row['min_lvl_rec'] >= 1 and row['max_lvl_rec'] <= 30:
            c += 1
    return c


def count_maps_low(x):
    c = 0
    x=x[['min_lvl_rec', 'max_lvl_rec']]
    for index, row in x.iterrows():
        if row['min_lvl_rec'] >= 20 and row['max_lvl_rec'] <= 50:
            c += 1
    return c


def count_maps_medium(x):
    c = 0
    x=x[['min_lvl_rec', 'max_lvl_rec']]
    for index, row in x.iterrows():
        if row['min_lvl_rec'] >= 40 and row['max_lvl_rec'] <= 70:
            c += 1
    return c


def count_maps_advanced(x):
    c = 0
    x=x[['min_lvl_rec', 'max_lvl_rec']]
    for index, row in x.iterrows():
        if row['min_lvl_rec'] >= 60 and row['max_lvl_rec'] <= 100:
            c += 1
    return c


def smooth_hours(x):
    if x < 0.2:
        return 0
    return x


def print_metrics(y_pred, y_test):
    prfs = precision_recall_fscore_support(y_pred, y_test)
    print("--- ING labels ---")
    print("precision: {:.3f}".format(prfs[0][0]))
    print("recall: {:.3f}".format(prfs[1][0]))
    print("fscore: {:.3f}".format(prfs[2][0]))
    print("--- ATC labels ---")
    print("precision: {:.3f}".format(prfs[0][1]))
    print("recall: {:.3f}".format(prfs[1][1]))
    print("fscore: {:.3f}".format(prfs[2][1]))
