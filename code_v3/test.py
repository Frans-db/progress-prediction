import numpy as np
import matplotlib.pyplot as plt


activities = {
    'Activity 1': np.array([1, 2, 5, 0, 0]),
    'Activity 2': np.array([1, 3, 6, 5, 0]),
    'Activity 3': np.array([1, 4, 7, 5, 6]),
}
COLORS = ['w', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'lime', 'darkblue']

def main():
    for activity_name in activities:
        activity = activities[activity_name]
        left = 0
        for action in activity:
            plt.barh(activity_name, 1, left=left, color=COLORS[action], height=0.5)
            left += 1

    plt.savefig('./test.png')

def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))

    data_cum = data.cumsum(axis=1)
    print(data_cum)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    # ax.set_xlim(0, np.sum(data, axis=1).max())

    ax.barh('test', 1, left=0, color='r', height=0.5)
    ax.barh('test', 1, left=1, color='g', height=0.5)

    ax.barh('test1', 1, left=0, color='g', height=0.5)
    ax.barh('test1', 1, left=1, color='r', height=0.5)
    # for i, (colname, color) in enumerate(zip(category_names, category_colors)):
    #     widths = data[:, i]
    #     widths = np.ones_like(widths)
    #     starts = np.arange(len(widths))
    #     print(widths, starts)
    #     ax.barh(labels, widths, left=starts, height=0.5,
    #             label=colname, color=color)

    # ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
    #           loc='lower left', fontsize='small')

    plt.savefig('./test.png')


main()