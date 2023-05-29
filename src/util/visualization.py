import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from scipy import interpolate
import numpy as np

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
    ax.axvline(mean)
    
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

def area_under_curve(data_x : list, data_y : list):
    """
    Args:
        data1: Index of experts
        data2: # of teams
        
    """

    fig, ax = plt.subplots(figsize = (10,6))

    # To plot a line graph of the data, interpolation can be used which creates a function based on the data points
    f = interpolate.interp1d(data_x, data_y, kind='linear') 
    xnew = np.arange(min(data_x), max(data_x), 0.001) # returns evenly spaced values from the data set
    ynew = f(xnew)
    ax.plot(xnew, ynew, color = 'red')

    mid_index = midP_calc(xnew, ynew) # helper function that calculates the mid point of the data set that divides the area equally

    
    # Fill left half of the area under the curve in blue
    ax.fill_between(xnew[ : mid_index + 1], ynew[ : mid_index + 1], color='blue', alpha=0.5)

    # Fill right half of the area under the curve in orange
    ax.fill_between(xnew[mid_index : ], ynew[mid_index : ], color='orange', alpha=0.5)
    
    # Scatter plot of data
    ax.scatter(x = data_x, y = data_y)


    
    # Titles for x and y axes
    plt.xlabel("X-Axis Values (Index of Experts)")
    plt.ylabel("Y-Axis Values (# Teams)")

    # Adding grid to display
    plt.grid()

    # Displays graph
    plt.show()

def midP_calc(x : list, y : list):
    mid_index = len(x) // 2
    
    # Dividing the area under the curve into 2 halves
    left_area = np.trapz(y[ : mid_index + 1], x[ : mid_index + 1])
    right_area = np.trapz(y[mid_index : ], x[mid_index : ])

    # Finding the mid point that divides it equally
    while(left_area != right_area):
        if(left_area > right_area):
            mid_index -= 1
        else:
            mid_index += 1
        left_area = round(np.trapz(y[ : mid_index + 1], x[ : mid_index + 1]), 3)
        right_area = round(np.trapz(y[mid_index : ], x[mid_index : ]), 3)
        
    return mid_index