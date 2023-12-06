import pickle as pkl
from scipy.sparse import find

import statistics
import scipy.sparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import interpolate
from collections import Counter



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

def plot_distribution(data1 : list, data2 : list, n : int):
    """
    Args:
        data1: Index of experts
        data2: Number of teams
        n: number of std dev to plot
    """
    # Calculations : mean, std dev, mean + 1 std dev, mean + 3 std dev
    mean = statistics.mean(data1)
    stdDev = statistics.stdev(data1)
    fig, ax = plt.subplots(figsize = (10, 6))
    # Plots data
    ax.scatter(x = data1, y = data2)
    # Plotting vertical line to indicate mean
    ax.axvline(mean, linewidth = 1.5, linestyle = "--", color = 'red')
    # Plotting vertical lines for mean + n * stdev
    for i in range(1, n + 1):
        ax.axvline(mean + i * stdDev)
    # Titles for x and y axes
    plt.xlabel("X-Axis Values")
    plt.ylabel("Y-Axis Values")
    # Adding grid to display
    plt.grid()
    # Displays graph
    plt.show()


def mid_calc(x : np.ndarray, y : np.ndarray)->int:
    """
    Helper function that calculates the middle point of the data set that divides the area equally
    Args:
        x: values of the x-axis
        y: values of the y-axis
    Returns:
        mid_index: index of the middle point
    """
    # Creating a start and end point for binary search 
    start = 0
    end = len(x) - 1
    mid_index = (start + end) // 2
    # Dividing the area under the curve into 2 halves
    left_area = round(np.trapz(y[ : mid_index + 1], x[ : mid_index + 1]), 3)
    right_area = round(np.trapz(y[mid_index : ], x[mid_index : ]), 3)
    # Finding the middle point that divides it equally
    while(abs(left_area - right_area) > 0.05 and start <= end):
        if (left_area > right_area):
            end = mid_index 
        elif (right_area > left_area):
            start = mid_index + 1
        else:
            return mid_index        
        mid_index = (start + end) // 2        
        left_area = round(np.trapz(y[ : mid_index + 1], x[ : mid_index + 1]), 3)
        right_area = round(np.trapz(y[mid_index : ], x[mid_index : ]), 3)
    return mid_index


def area_under_curve(data_x: list, data_y: list, xlabel: str, ylabel: str, lcolor: str='green', rcolor: str='orange', show_plot: bool=True)->float:
    """
    Args:
        data1: Index of experts
        data2: # of teams
        xlabel: label for x axis
        ylabel: label for y axis
    """
    fig, ax = plt.subplots(figsize = (2,2))
    plt.rcParams.update({'font.family': 'Consolas'})
    # To plot a line graph of the data, interpolation can be used which creates a function based on the data points
    f = interpolate.interp1d(data_x, data_y, kind='linear') 
    xnew = np.arange(min(data_x), max(data_x), 0.001) # returns evenly spaced values from the data set
    ynew = f(xnew)
    ax.plot(xnew, ynew, color = 'red')
    mid_index = mid_calc(xnew, ynew) # helper function that calculates the mid point of the data set that divides the area equally
    # Fill left half of the area under the curve in parameter passed for lcolor
    ax.fill_between(xnew[ : mid_index + 1], ynew[ : mid_index + 1], color=lcolor, alpha=0.3)
    # Fill right half of the area under the curve in parameter passed for rcolor
    ax.fill_between(xnew[mid_index : ], ynew[mid_index : ], color=rcolor, alpha=0.5)
    # Scatter plot of data
    ax.scatter(x = data_x, y = data_y)
    # Titles for x and y axes
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Graph appearance settings 
    ax.grid(True, color="#93a1a1", alpha=0.3)
    ax.minorticks_off()
    ax.xaxis.get_label().set_size(12)
    ax.yaxis.get_label().set_size(12)
    plt.legend(fontsize='small')
    ax.set_facecolor('whitesmoke')

    # Displays graph
    if show_plot:
        plt.show()
        fig.savefig(f'auc-avg.pdf', dpi=200, bbox_inches='tight')
    return ynew[mid_index]

def attribute_distribution_plot(teamsvecs: scipy.sparse.lil_matrix, index_att: pd.DataFrame, plot_title: str, att: str):
    """
    Args:
        teamsvecs: sparse matrix of teamsvecs
        index_att: mapping from indexes to genders as pandas dataframe
        plot_title: title for the plot
    """
    if att == 'gender':
        protected = index_att.loc[index_att[att] == False, 'Unnamed: 0'].tolist()
        nonprotected = index_att.loc[index_att[att] == True, 'Unnamed: 0'].tolist()
        legendp, legendnp = 'female', 'male'
    else:
        protected = index_att.loc[index_att[att] == False, 'memberidx'].tolist()
        nonprotected = index_att.loc[index_att[att] == True, 'memberidx'].tolist()
        legendp, legendnp = 'nonpopular', 'popular'

    nteams_id_male = teamsvecs['member'][:, nonprotected].sum(axis=0)
    nteams_id_female = teamsvecs['member'][:, protected].sum(axis=0)

    stats = dict()
    nmembers_nteams_male = Counter(nteams_id_male.A1.astype(int))
    nmembers_nteams_female = Counter(nteams_id_female.A1.astype(int))
    stats[legendnp] = {k: v for k, v in sorted(nmembers_nteams_male.items(), key=lambda item: item[1], reverse=True)}
    stats[legendp] = {k: v for k, v in sorted(nmembers_nteams_female.items(), key=lambda item: item[1], reverse=True)}

    fig = plt.figure(figsize=(2, 2))
    plt.rcParams.update({'font.family': 'Consolas'})
    ax = fig.add_subplot(1, 1, 1)
    for k, v in stats.items():
        marker_markeredgecolor = ('x', 'b') if k == legendnp else ('o', 'r')
        ax.plot(*zip(*stats[k].items()), marker=marker_markeredgecolor[0], label=k.lower(), linestyle='None', markeredgecolor=marker_markeredgecolor[1], markerfacecolor='none')

    ax.set_xlabel('#teams')
    ax.set_ylabel('#members')
    ax.grid(True, color="#93a1a1", alpha=0.3)
    ax.minorticks_off()
    ax.xaxis.get_label().set_size(12)
    ax.yaxis.get_label().set_size(12)
    plt.legend(fontsize='small')
    ax.set_title(plot_title, fontsize=11)
    ax.set_facecolor('whitesmoke')
    fig.savefig(f'{plot_title}_{att}_nmembers_nteams.pdf', dpi=200, bbox_inches='tight')
    plt.show()



def graph_members_with_most_teams_topK(teamsvecsFile: str, k: int):
    """
    Generates a graph that plots the members from most amount of teams to least amount of teams.
    Will also find the area under the curve and seperate it into two equal portions
    Args:
        teamsvecsFile: file location for the teamsvecs.pkl on the results
        k: top-k highest number of teams to graph
    """
    data_x = [] 
    data_y = None
    with open(teamsvecsFile, "rb") as f:
        opeNTF_out = pkl.load(f)

        ROWS = opeNTF_out['member'].get_shape()[0]
        COLS = opeNTF_out['member'].get_shape()[1]

        if(not k): k = COLS # if no k is given, plot all teams

        data_y = [0] * COLS

        # For each team recorded:
        for j in range(0, ROWS):
            row = (opeNTF_out["member"].getrow(j)) 
            # Find all the non-zero entries: (what the [1] is for)
            for i in find(row)[1]: data_y[i] += 1 

        # Make x axis 0 to k:
        for i in range(0, k): data_x.append(i)

        # Get top-k "y" values
        data_y.sort(reverse=True)
        data_y = data_y[:k]

    # Plot area_under_curve with the obtained data:
    area_under_curve(data_x=data_x, data_y=data_y, xlabel="expert-idx", ylabel="#teams")
