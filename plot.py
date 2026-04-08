import numpy as np
import matplotlib.pyplot as plt
from imgcat import imgcat

def plot_words_pyplot(x_2d, labels, max_points_per_word=None):
    '''plot 2d points per word using matplotlib with well-distributed 
    color/marker pairs.
    x_2d                   array (n_samples, 2)
    labels                 list of word labels
    max_points_per_word    optional int to subsample per word
    '''

    unique_words = list(dict.fromkeys(labels))

    colors = list(plt.cm.tab20.colors)
    markers = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', '<', '>', 'h', 'p']

    styles = make_styles(colors, markers)

    plt.figure(figsize=(12, 9))

    for i, word in enumerate(unique_words):
        idx = [j for j, label in enumerate(labels) if label == word]

        if max_points_per_word and len(idx) > max_points_per_word:
            idx = np.random.choice(idx, max_points_per_word, replace=False)

        xs = x_2d[idx, 0]
        ys = x_2d[idx, 1]

        color, marker = styles[i % len(styles)]

        plt.scatter(
            xs,
            ys,
            c=[color],
            marker=marker,
            s=14,
            alpha=0.75,
            label=word
        )

    plt.title('t-SNE of word vectors')
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')

    if len(unique_words) <= 15:
        plt.legend()
    else:
        plt.legend(fontsize=8, ncols=2)

    plt.tight_layout()
    plt.savefig('_plot.png')
    imgcat(open('_plot.png'))


def make_styles(colors, markers):
    '''make a distributed list of color-marker combinations.
    colors                 list of matplotlib colors
    markers                list of matplotlib markers
    '''
    n_colors = len(colors)
    n_markers = len(markers)

    step = choose_step(n_markers)

    styles = []
    for color_index in range(n_colors):
        for k in range(n_markers):
            marker_index = (color_index + k * step) % n_markers
            styles.append((colors[color_index], markers[marker_index]))
    return styles


def choose_step(n):
    '''choose a step size coprime with n.
    n                      cycle length
    '''
    for step in [5, 7, 11, 13, 17, 19, 23]:
        if np.gcd(step, n) == 1:
            return step
    return 1
