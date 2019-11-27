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
