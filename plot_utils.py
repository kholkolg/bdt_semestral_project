import pandas as pd
from os import path, listdir
import numpy as np
from matplotlib import pyplot as plt
from typing import List

from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.ticker as mtick
from matplotlib.ticker import StrMethodFormatter





SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
FIGSIZE1 = (16, 8)

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


pic_dir = '/home/olga/Documents/BDT/results/pics2'
YEAR = 2015




def read_df(dir:str, columns:List[str] = None, sort_column:str = None)->pd.DataFrame:

    # read data from multiple files with results into dataframe

    files = map(lambda  file: path.join(dir, file), listdir(dir))
    df: pd.DataFrame = pd.concat(pd.read_csv(file, index_col=None, header=None) for file in files)

    if columns:
        df.rename({i:columns[i] for i in range(len(columns))}, axis=1, inplace=True)

    if sort_column:
        df.sort_values(sort_column, inplace=True)

    print('Dataframe loaded. Shape %s. Columns: %s' % (df.shape, str(df.columns)))
    return df


def plot_column(df:pd.DataFrame, column:str, ylabel:str, title=None, filename=None):

    # plot one column from dataframe against time on x-axis

    fig, ax = plt.subplots(1, figsize=FIGSIZE1)

    ax.plot(df.index.values, df[column].values,  color='#86bf91', zorder=2)

    for tick in ax.get_yticks():
        ax.axhline(y=tick, linestyle='dashed', alpha=0.6, color='#dddddd', zorder=1)
    ax.set_ylabel(ylabel, labelpad=20)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%d/%m"))

    if title:
        ax.set_title(title)

    if filename:
        plt.savefig(path.join(pic_dir, filename), dpi=300, orientation='landscape', bbox_inches='tight')

    plt.show()


def plot_heatmap(df, values_col, title=None, filename=None):

    # heatmap for weekdays and day times

    result = df.pivot(index='Hour', columns='Week Day', values=values_col)
    sns.heatmap(result, annot=True, fmt="g", linewidths=0.1, cbar=False)

    ax = plt.gca()
    if title:
        ax.set_title(title)

    if filename:
        plt.savefig(path.join(pic_dir, filename), dpi=300, orientation='landscape', bbox_inches='tight')

    plt.show()


def plot_weather(df, columns):


    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24, 8))
    for i, var in enumerate(columns[1:]):
        r = int(i%2)
        c = int(i/2)
        ax = axes[r, c]
        ax.plot(df.index.values, df[var].values, color='#86bf91', zorder=2)

        for tick in ax.get_yticks():
            ax.axhline(y=tick, linestyle='dashed', alpha=0.6, color='#dddddd', zorder=1)

        ax.set_ylabel(var, labelpad=10)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%d/%m"))

        #         df.plot(ax=ax, x='date', y = var,  title=var, colormap='jet')
        # axes[r][c].set_xlabel(" ")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%d/%m"))

    plt.savefig(path.join(pic_dir, 'weather_means_sub.png'), dpi=300, orientation='landscape')
    plt.show()



def plot_rdd_histogram(data, title, xlabel, ylabel, filename=None, xticks=None):

    bin_sides, bin_counts = data

    N = len(bin_counts)
    ind = np.arange(N)
    width = 0.9

    fig, ax = plt.subplots(1, figsize=FIGSIZE1)
    ax.bar(ind+0.5, bin_counts, width,  color='#86bf91', zorder=2)
    vals = ax.get_yticks()
    for tick in vals:
        ax.axhline(y=tick, linestyle='dashed', alpha=0.6, color='#dddddd', zorder=1)
    ax.set_ylabel(ylabel, labelpad=20)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))
    ax.set_xlabel(xlabel, labelpad=20)

    if title:
        ax.set_title(title)

    if xticks:
        plt.xticks(np.arange(N + 1), np.round(np.array(bin_sides)), rotation='vertical')

    if filename:
        plt.savefig(path.join(pic_dir, filename), dpi=300, orientation='landscape', bbox_inches='tight')

    plt.show()

