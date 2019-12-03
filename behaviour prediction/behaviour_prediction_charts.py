import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

input_dir = '../warcraft-avatar-history/'
output_dir = '../warcraft-avatar-history/'
save_dir = 'results/'


def main():
    wow = pd.read_csv(input_dir + 'wowa_data_labeled.csv')

    wow = wow[['lvl_end', 'behaviour', 'evolution', 'n_maps', 'time_hours', 'n_pvp']]

    classes = ['killer', 'socializer', 'achiever', 'explorer']
    n_killer = wow[wow['behaviour'] == 'killer']['behaviour'].count()
    n_socializer = wow[wow['behaviour'] == 'socializer']['behaviour'].count()
    n_achiever = wow[wow['behaviour'] == 'achiever']['behaviour'].count()
    n_explorer = wow[wow['behaviour'] == 'explorer']['behaviour'].count()
    sizes = [n_killer, n_socializer, n_achiever, n_explorer]

    # draw pie chart
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=classes, autopct='%1.1f%%', startangle=90)
    # draw circle
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')
    plt.tight_layout()
    plt.legend(loc='upper left')
    plt.savefig(save_dir + "pie_chart")

    # encode behaviours type with numerical values
    labels = []
    for index, row in wow.iterrows():
        if row['behaviour'] in classes:
            labels.append(classes.index(row['behaviour']))
        else:
            labels.append(-1)
    labels = np.asarray(labels)
    wow["behaviour_num"] = pd.Series(labels)
    fig1 = sns.lmplot(x='n_maps', y='behaviour_num', hue='behaviour', data=wow, fit_reg=False, y_jitter=0.25)
    fig2 = sns.lmplot(x='lvl_end', y='behaviour_num', hue='behaviour', data=wow, fit_reg=False, y_jitter=0.25)
    fig3 = sns.lmplot(x='evolution', y='behaviour_num', hue='behaviour', data=wow, fit_reg=False, y_jitter=0.25)
    fig4 = sns.lmplot(x='n_pvp', y='behaviour_num', hue='behaviour', data=wow, fit_reg=False, y_jitter=0.25)
    fig1.savefig(save_dir + "fig1.png")
    fig2.savefig(save_dir + "fig2.png")
    fig3.savefig(save_dir + "fig3.png")
    fig4.savefig(save_dir + "fig4.png")

    plt.show()


if __name__ == "__main__":
    main()
