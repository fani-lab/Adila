import matplotlib.pyplot as plt
import seaborn as sns


def bubble_plot(fairness: list, utility: list, runtime: list, figsize: list = [14, 8], xlabel: str = "Fairness Metric", ylabel: str = "Utility Metric", save: bool = False):
    """
    Args:
        fairness: fairness metric values
        utility: utility metric values
        runtime: runtime of each sample
        figsize: figure size for plot
        xlabel: label for the x-axis on the plot
        ylabel: label for the y-axis on the plot
        save: flag to determine saving the plot
    """

    plt.rcParams['figure.figsize'] = figsize
    sns.set_style("darkgrid")  # used just for a darker background with grids (not required)
    # Plots data using seaborn
    sns.scatterplot(x=fairness, y=utility, size=runtime, sizes=(min(runtime), max(runtime)), alpha=0.5)
    # Titles for x and y axes
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Brings legend out of the graph region
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=10)

    if save:
        plt.savefig(f'{xlabel} vs {ylabel}.png')
    # Displays graph
    plt.show()
