import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from jinja2 import Template
from IPython.display import HTML
import base64
from io import BytesIO
import pandas as pd

from wordcloud import WordCloud


from ..datahub import DataHub
from ..dtype import *

from .tables import *


"""""" # NEW IMPORTS
from nltk.tokenize import word_tokenize
import nltk

if not nltk.downloader.Downloader().is_installed('punkt'):
    # If not, download the 'punkt' package
    nltk.download('punkt')
else:
    pass
    #print("Package 'punkt' is already downloaded.")
""""""


"""
This module implements the visualization for the plot(df) function.
"""


        
# def hist_plot_thumb_mini(data,cfg, num_bins=None):

#     # Exclude NaN values
#     data_non_nan = data.dropna()
    
#     unique = data.nunique()
#     fig_histogram, ax_histogram = plt.subplots(figsize=(3, 0.4))

#     if num_bins is None:
#         num_bins = min(unique, 30)  # Adjust 40 to your preference or remove for automatic calculation

#     ax_histogram.hist(data_non_nan, bins=num_bins, color=cfg['mini_hist_color'], edgecolor='white') #color='#06c'
#     ax_histogram.yaxis.set_visible(False)
#     ax_histogram.set_yscale('log')  # Set logarithmic scale on the y-axis TO HANDLE VISUALIZE OUTLIERS!!!
#     ax_histogram.xaxis.set_visible(False)
#     ax_histogram.set_frame_on(False)
#     plt.grid(False)

#     # Save each figure to a buffer
#     buffer_histogram = BytesIO()
#     fig_histogram.savefig(buffer_histogram, format='png')
#     image_histogram_thumb_mini = base64.b64encode(buffer_histogram.getvalue()).decode('utf-8')
#     plt.close(fig_histogram)

#     return image_histogram_thumb_mini

""" POSSIBLE IMPORVEMENT calling plt.hist instad of plt.subplots """

def hist_plot_thumb_mini(data, cfg, num_bins=None):
    # Exclude NaN values
    data_non_nan = data.dropna()

    unique = data.nunique()

    if num_bins is None:
        num_bins = min(unique, 30)

    plt.figure(figsize=(3, 0.4))
    plt.hist(data_non_nan, bins=num_bins, color=cfg['mini_hist_color'], edgecolor='white')
    plt.yscale('log')
    plt.axis('off')
    plt.grid(False)

    buffer_histogram = BytesIO()
    plt.savefig(buffer_histogram, format='png')
    image_histogram_thumb_mini = base64.b64encode(buffer_histogram.getvalue()).decode('utf-8')
    plt.close()

    return image_histogram_thumb_mini
    
   
def hist_plot(data,cfg):
        fig_histogram, ax_histogram = plt.subplots(figsize=(6, 4))
        ax_histogram.hist(data, bins=30, color='skyblue', edgecolor='black')
        ax_histogram.set_title('Histogram')
        ax_histogram.set_xlabel('Value')
        ax_histogram.set_ylabel('Frequency')
    
        # Save each figure to a buffer
        buffer_histogram = BytesIO()
        fig_histogram.savefig(buffer_histogram, format='png')
        image_histogram = base64.b64encode(buffer_histogram.getvalue()).decode('utf-8')
        plt.close(fig_histogram)

        return image_histogram
'''testing hist + kde '''

# def hist_plot_thumb(data,cfg, name=None):
#     # Create a histogram
#     fig_histogram, ax_histogram = plt.subplots(figsize=(6, 4))
#     ax_histogram.hist(data, bins=30, color=cfg['hist_color'], edgecolor='white', density=True) #'skyblue'
    
#     # Add a kernel density plot
#     sns.kdeplot(data, color=cfg['hist_kde_color'], ax=ax_histogram, warn_singular=False) #'pink'
    
#     ax_histogram.tick_params(axis='both', colors='#a9a9a9')
#     ax_histogram.set_xlabel('Value', color='#a9a9a9')
#     ax_histogram.set_ylabel('Frequency', color='#a9a9a9')
#     if name is not None:
#         ax_histogram.set_title(name)
#     ax_histogram.set_frame_on(False)
#     plt.grid(False)

#     # Save the figure to a buffer
#     buffer_histogram = BytesIO()
#     fig_histogram.savefig(buffer_histogram, format='png')
#     image_histogram_thumb = base64.b64encode(buffer_histogram.getvalue()).decode('utf-8')
#     plt.close(fig_histogram)

#     return image_histogram_thumb

    
def kde_plot(data):
        fig_kde, ax_kde = plt.subplots(figsize=(6, 4))
        sns.kdeplot(data, color='salmon')
        ax_kde.set_title('KDE Plot')
        ax_kde.set_xlabel('Value')
        ax_kde.set_ylabel('Density')

        # Save figure to a buffer
        buffer_kde = BytesIO()
        fig_kde.savefig(buffer_kde, format='png')
        image_kde = base64.b64encode(buffer_kde.getvalue()).decode('utf-8')
        plt.close(fig_kde)

        return image_kde
#axxx    cfg['mini_hist_color']
def qq_plot(data, cfg):
    # Exclude NaN values
    data_non_nan = data.dropna()

    # Set the plot border to light grey
    fig_qq, ax_qq = plt.subplots(figsize=(6, 4))
    
    qq_line = stats.probplot(data_non_nan, plot=ax_qq, fit=True, rvalue=True)
    
    # Customize plot colors based on cfg dictionary
    line_color = cfg.get('blue', 'blue')
    marker_color = cfg.get('mini_hist_color', 'red')
    
    # Customize line and marker colors
    ax_qq.get_lines()[0].set_markerfacecolor(marker_color)
    ax_qq.get_lines()[0].set_markeredgecolor(marker_color)
    ax_qq.get_lines()[0].set_color(line_color)
    
    # Customize text colors
    title_color = cfg.get('title_color', '#a9a9a9')
    xlabel_color = cfg.get('xlabel_color', '#a9a9a9')
    ylabel_color = cfg.get('ylabel_color', '#a9a9a9')
    tick_color = cfg.get('tick_color', '#a9a9a9')

    ax_qq.set_title('Normal Q-Q Plot', color=title_color)
    ax_qq.set_xlabel('Theoretical Quantiles', color=xlabel_color)
    ax_qq.set_ylabel('Ordered Values', color=ylabel_color)

    # Set spines color to light grey
    for spine in ax_qq.spines.values():
        spine.set_edgecolor('#d3d3d3')

    # Set xticks and yticks color to #a9a9a9
    ax_qq.tick_params(axis='x', colors=tick_color)
    ax_qq.tick_params(axis='y', colors=tick_color)
    ax_qq.grid(color=tick_color, linestyle=':', linewidth=0.5, alpha=0.5)

    # Save figure to a buffer
    buffer_qq = BytesIO()
    fig_qq.savefig(buffer_qq, format='png')
    image_qq = base64.b64encode(buffer_qq.getvalue()).decode('utf-8')
    plt.close(fig_qq)

    return image_qq


def box_plot(data):
        fig_box, ax_box = plt.subplots(figsize=(6, 4))
        sns.boxplot(data, color='lightgreen')
        ax_box.set_title('Box Plot')
        ax_box.set_xlabel('Value')

        # Save figure to a buffer
        buffer_box = BytesIO()
        fig_box.savefig(buffer_box, format='png')
        image_box = base64.b64encode(buffer_box.getvalue()).decode('utf-8')
        plt.close(fig_box)

        return image_box

def bar_chart(data):
    """
    Generate a horizontal bar chart.

    Parameters:
    - data (pandas.Series or pandas.DataFrame): Data for the bar chart.

    Returns:
    - str: Base64-encoded PNG image of the bar chart.
    """
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()  # Convert DataFrame to Series if needed

    value_counts = data.value_counts(normalize=True) * 100
    
    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
    value_counts.plot(kind='barh', color='skyblue', edgecolor='black', ax=ax_bar)
    
    ax_bar.set_title('Horizontal Bar Chart')
    ax_bar.set_xlabel('Count')
    ax_bar.set_ylabel('Category')

    # Set explicit ticks and add percentage signs
    ticks = ax_bar.get_xticks()
    tick_labels = [f'{tick:.2f}%' for tick in ticks]
    ax_bar.set_xticks(ticks)
    ax_bar.set_xticklabels(tick_labels)

    
    # Save figure to a buffer
    buffer_bar = BytesIO()
    fig_bar.savefig(buffer_bar, format='png')
    image_bar = base64.b64encode(buffer_bar.getvalue()).decode('utf-8')
    plt.close(fig_bar)

    return image_bar

###########################################################################################################################
######################################### BIVARIATE CHARTS ###########################################################
###########################################################################################################################

"""BIVARIATE"""




def scatter_plot(x, y):
    # Create a DataFrame from the data
    df = pd.DataFrame({'X': x, 'Y': y})

    # Create a scatter plot
    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='X', y='Y', data=df, color='coral', alpha=0.6, ax=ax_scatter)

    # Set plot title and labels
    ax_scatter.set_title(f"{x.name} vs {y.name}", color='#a9a9a9')
    ax_scatter.set_xlabel(x.name, color='#a9a9a9')
    ax_scatter.set_ylabel(y.name, color='#a9a9a9')

    # Add grid
    ax_scatter.grid(color='#a9a9a9', linestyle='dashed', linewidth=0.5)

    # Set border color of the plot to light grey
    for spine in ax_scatter.spines.values():
        spine.set_edgecolor('lightgrey')

    # Set font color for all text in the chart
    for text in ax_scatter.texts + ax_scatter.get_xticklabels() + ax_scatter.get_yticklabels():
        text.set_color('#a9a9a9')

    # Save figure to a buffer
    buffer_scatter = BytesIO()
    fig_scatter.savefig(buffer_scatter, format='png')
    image_scatter = base64.b64encode(buffer_scatter.getvalue()).decode('utf-8')
    plt.close(fig_scatter)

    return image_scatter

def hexbin_plot(x, y):
    # Create a hexbin plot
    fig_hexbin, ax_hexbin = plt.subplots(figsize=(8, 6))
    hb = ax_hexbin.hexbin(x, y, gridsize=30, cmap='Blues')

    # Set plot title and labels
    ax_hexbin.set_title(f"{x.name} vs {y.name}", color='#a9a9a9')
    ax_hexbin.set_xlabel(x.name, color='#a9a9a9')
    ax_hexbin.set_ylabel(y.name, color='#a9a9a9')

    # Add colorbar
    cbar = plt.colorbar(hb)
    cbar.set_label('Count', color='#a9a9a9')

    # Set font color for ticks on both x and y axes
    ax_hexbin.tick_params(axis='both', colors='#a9a9a9')

    # Set border color of the plot to light grey
    for spine in ax_hexbin.spines.values():
        spine.set_edgecolor('lightgrey')

    # Save figure to a buffer
    buffer_hexbin = BytesIO()
    fig_hexbin.savefig(buffer_hexbin, format='png')
    image_hexbin = base64.b64encode(buffer_hexbin.getvalue()).decode('utf-8')
    plt.close(fig_hexbin)

    return image_hexbin


# def scatter_plot(x, y):
#     fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
#     ax_scatter.scatter(x, y, color='coral', alpha=0.6)
#     ax_scatter.set_title('Scatter Plot')
#     ax_scatter.set_xlabel('X')
#     ax_scatter.set_ylabel('Y')

#     # # Add grid
#     # ax_scatter.grid(True)

#     # Save figure to a buffer
#     buffer_scatter = BytesIO()
#     fig_scatter.savefig(buffer_scatter, format='png')
#     image_scatter = base64.b64encode(buffer_scatter.getvalue()).decode('utf-8')
#     plt.close(fig_scatter)

#     return image_scatter

# def hexbin_plot(x, y):
#     fig_hexbin, ax_hexbin = plt.subplots(figsize=(8, 6))
#     ax_hexbin.hexbin(x, y, gridsize=30, cmap='Blues')
#     ax_hexbin.set_title('Hexbin Plot')
#     ax_hexbin.set_xlabel('X')
#     ax_hexbin.set_ylabel('Y')

#     # Save figure to a buffer
#     buffer_hexbin = BytesIO()
#     fig_hexbin.savefig(buffer_hexbin, format='png')
#     image_hexbin = base64.b64encode(buffer_hexbin.getvalue()).decode('utf-8')
#     plt.close(fig_hexbin)

#     return image_hexbin
    
""" doesnt work check"""

# def pair_plot(dataframe):
#     fig_pair, ax_pair = plt.subplots(figsize=(8, 8))
#     sns.pairplot(dataframe, palette='viridis', diag_kind='kde', markers='o')
#     ax_pair.set_title('Pair Plot')

#     # Save figure to a buffer
#     buffer_pair = BytesIO()
#     fig_pair.savefig(buffer_pair, format='png')
#     image_pair = base64.b64encode(buffer_pair.getvalue()).decode('utf-8')
#     plt.close(fig_pair)

#     return image_pair

# def correlation_heatmap(dataframe):
#     fig_heatmap, ax_heatmap = plt.subplots(figsize=(8, 6))
#     sns.heatmap(dataframe.corr(), annot=True, cmap='coolwarm', fmt=".2f")
#     ax_heatmap.set_title('Correlation Heatmap')

#     # Save figure to a buffer
#     buffer_heatmap = BytesIO()
#     fig_heatmap.savefig(buffer_heatmap, format='png')
#     image_heatmap = base64.b64encode(buffer_heatmap.getvalue()).decode('utf-8')
#     plt.close(fig_heatmap)

#     return image_heatmap
#ax
# def box_plot_con(x, y):

#     # Ensure x and y are numeric (convert if necessary)
#     fig_box, ax_box = plt.subplots(figsize=(8, 6))
#     sns.boxplot(x=x, y=y, ax=ax_box)
    
#     ax_box.set_title('Box Plot: Numerical vs Categorical')
#     ax_box.set_xlabel('Categorical Variable')
#     ax_box.set_ylabel('Numerical Variable')

#     # Save figure to a buffer
#     buffer_box = BytesIO()
#     fig_box.savefig(buffer_box, format='png')
#     image_box_con = base64.b64encode(buffer_box.getvalue()).decode('utf-8')
#     plt.close(fig_box)

#     return image_box_con

def box_plot_con(x, y):
    # Create a DataFrame from the data
    df = pd.DataFrame({'X': x, 'Y': y})

    # Create a box plot
    fig_box, ax_box = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='X', y='Y', data=df, hue='X', palette='Set3', ax=ax_box, dodge=False, legend=True)

    # Set plot title and labels
    ax_box.set_title(f"{x.name} vs {y.name}", color='#a9a9a9')
    ax_box.set_xlabel(x.name, color='#a9a9a9')
    ax_box.set_ylabel(y.name, color='#a9a9a9')

    # Set x-axis ticks and labels
    x_ticks = np.arange(len(ax_box.get_xticklabels()))
    ax_box.set_xticks(x_ticks)
    
    # Crop x-axis labels if longer than 20 symbols
    labels = [text.get_text() if len(text.get_text()) <= 10 else text.get_text()[:10] + '...' for text in ax_box.get_xticklabels()]
    ax_box.set_xticklabels(labels, ha='center')

    # Set font color for all text in the chart
    for text in ax_box.texts + ax_box.get_xticklabels() + ax_box.get_yticklabels():
        text.set_color('#a9a9a9')

    ax_box.xaxis.grid(color='#a9a9a9', linestyle='dashed', linewidth=0.5)
    ax_box.yaxis.grid(color='#a9a9a9', linestyle='dashed', linewidth=0.5)

    # Set the plot border color to light grey
    ax_box.spines['top'].set_color('#d3d3d3')
    ax_box.spines['bottom'].set_color('#d3d3d3')
    ax_box.spines['left'].set_color('#d3d3d3')
    ax_box.spines['right'].set_color('#d3d3d3')

    # Set font color for legend
    legend = ax_box.legend(title=y.name)
    legend.get_title().set_color('#a9a9a9')

    # Save figure to a buffer
    buffer_box = BytesIO()
    fig_box.savefig(buffer_box, format='png')
    image_box = base64.b64encode(buffer_box.getvalue()).decode('utf-8')
    plt.close(fig_box)

    return image_box

def violin_plot(x, y):
    # Create a DataFrame from the data
    df = pd.DataFrame({'X': x, 'Y': y})

    # Create a violin plot
    fig_violin, ax_violin = plt.subplots(figsize=(8, 6))
    sns.violinplot(x='X', y='Y', data=df, hue='X', palette='Set3', ax=ax_violin, inner='quartile', split=False, legend=False)

    # Set plot title and labels
    ax_violin.set_title(f"{x.name} vs {y.name}", color='#a9a9a9')
    ax_violin.set_xlabel(x.name, color='#a9a9a9')
    ax_violin.set_ylabel(y.name, color='#a9a9a9')

    # Set x-axis ticks and labels
    x_ticks = np.arange(len(ax_violin.get_xticklabels()))
    ax_violin.set_xticks(x_ticks)
    
    # Crop x-axis labels if longer than 20 symbols
    labels = [text.get_text() if len(text.get_text()) <= 10 else text.get_text()[:10] + '...' for text in ax_violin.get_xticklabels()]
    ax_violin.set_xticklabels(labels, ha='center')

    # Set font color for all text in the chart
    for text in ax_violin.texts + ax_violin.get_xticklabels() + ax_violin.get_yticklabels():
        text.set_color('#a9a9a9')

    ax_violin.xaxis.grid(color='#a9a9a9', linestyle='dashed', linewidth=0.5)
    ax_violin.yaxis.grid(color='#a9a9a9', linestyle='dashed', linewidth=0.5)

    # Set the plot border color to light grey
    ax_violin.spines['top'].set_color('#d3d3d3')
    ax_violin.spines['bottom'].set_color('#d3d3d3')
    ax_violin.spines['left'].set_color('#d3d3d3')
    ax_violin.spines['right'].set_color('#d3d3d3')

    # Save figure to a buffer
    buffer_violin = BytesIO()
    fig_violin.savefig(buffer_violin, format='png')
    image_violin = base64.b64encode(buffer_violin.getvalue()).decode('utf-8')
    plt.close(fig_violin)

    return image_violin

# def line_chart(x, y):
#     fig_line, ax_line = plt.subplots(figsize=(6, 4))
#     ax_line.plot(x, y, marker='o', color='purple', linestyle='-', linewidth=2, markersize=8)
#     ax_line.set_title('Line Chart')
#     ax_line.set_xlabel('X')
#     ax_line.set_ylabel('Y')

#     # Save figure to a buffer
#     buffer_line = BytesIO()
#     fig_line.savefig(buffer_line, format='png')
#     image_line = base64.b64encode(buffer_line.getvalue()).decode('utf-8')
#     plt.close(fig_line)

#     return image_line


def line_chart(x, y):
    # Create a DataFrame
    df = pd.DataFrame({'Age': y, 'Sex': x})
    
    # Plot the value_counts of age separated by sex
    plt.figure(figsize=(10, 6))
    df.groupby(['Age', 'Sex']).size().unstack().plot(kind='line', linewidth=1)
    plt.title('Value Counts of Age Separated by Sex')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.legend(title='Sex')
    plt.grid(True)

    # Save figure to a buffer
    buffer_age_distribution = BytesIO()
    plt.savefig(buffer_age_distribution, format='png')
    image_age_distribution = base64.b64encode(buffer_age_distribution.getvalue()).decode('utf-8')
    plt.close()

    return image_age_distribution


def heatmap_plot(x, y):
    # Create a DataFrame from the categorical variables
    df = pd.DataFrame({'X': x, 'Y': y})

    # Create a pivot table
    pivot_table = df.pivot_table(index='Y', columns='X', aggfunc=len, fill_value=0)

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='g')

    # Set font color for axis labels and title
    plt.title('Heatmap Plot', color='#a9a9a9')
    plt.xlabel((x.name).upper(), color='#a9a9a9')
    plt.ylabel((y.name).upper(), color='#a9a9a9')

    # Set font color for annotation text
    # for text in plt.gca().texts:
    #     text.set_color('#a9a9a9')

    # Set font color for ticks
    plt.xticks(color='#a9a9a9')
    plt.yticks(color='#a9a9a9')

    # Save figure to a buffer
    buffer_heatmap = BytesIO()
    plt.savefig(buffer_heatmap, format='png')
    image_heatmap = base64.b64encode(buffer_heatmap.getvalue()).decode('utf-8')
    plt.close()

    return image_heatmap

    

def stacked_bar_chart_percentage(x, y, width, height):
    # Create a DataFrame from the categorical variables
    df = pd.DataFrame({'x': x, 'y': y})

    # Create a cross-tabulation
    cross_tab = pd.crosstab(df['x'], df['y'])

    # Normalize the data to get percentages
    cross_tab_percentage = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100

    # Convert index to strings if they are integers
    cross_tab_percentage.index = cross_tab_percentage.index.astype(str)

    # Crop x-axis ticks if longer than 20 symbols
    labels = [text if len(text) <= 10 else text[:10] + '...' for text in cross_tab_percentage.index]

    # Create a stacked bar chart with percentages
    fig, ax = plt.subplots(figsize=(width, height))
    cross_tab_percentage.plot(kind='bar', stacked=True, colormap='Set3', ax=ax)

    # Rotate x ticks 45 degrees
    ax.set_xticklabels(labels, ha='center', rotation=0)

    # Set font color to a9a9a9 for all text in the chart
    for text in ax.texts + ax.get_xticklabels() + ax.get_yticklabels():
        text.set_color('#a9a9a9')

    # Set frame and grid color to a9a9a9
    for spine in ax.spines.values():
        spine.set_edgecolor('#d3d3d3')

    ax.set_title('Stacked Bar Chart (Percentage)', color='#a9a9a9')
    ax.set_xlabel(x.name, color='#a9a9a9')
    ax.set_ylabel('Percentage', color='#a9a9a9')

    # Set font color for legend
    legend = ax.legend(title=y.name)
    legend.get_title().set_color('#a9a9a9')

    # Save figure to a buffer
    buffer_stacked_bar_percentage = BytesIO()
    fig.savefig(buffer_stacked_bar_percentage, format='png')
    image_stacked_bar_percentage = base64.b64encode(buffer_stacked_bar_percentage.getvalue()).decode('utf-8')
    plt.close()

    return image_stacked_bar_percentage





# def stacked_bar_chart_percentage(x, y, width, height):
#     # Create a DataFrame from the categorical variables
#     df = pd.DataFrame({'X': x, 'Y': y})

#     # Create a cross-tabulation
#     cross_tab = pd.crosstab(df['X'], df['Y'])

#     # Normalize the data to get percentages
#     cross_tab_percentage = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100

#     # Crop x-axis ticks if longer than 20 symbols
#     labels = [text if len(text) <= 10 else text[:10] + '...' for text in cross_tab_percentage.index]

#     # Create a stacked bar chart with percentages
#     fig, ax = plt.subplots(figsize=(width, height))
#     cross_tab_percentage.plot(kind='bar', stacked=True, colormap='Set3', ax=ax)

#     # Rotate x ticks 45 degrees
#     ax.set_xticklabels(labels, ha='center', rotation=0)

#     # Set font color to a9a9a9 for all text in the chart
#     for text in ax.texts + ax.get_xticklabels() + ax.get_yticklabels():
#         text.set_color('#a9a9a9')

#     # Set frame and grid color to a9a9a9
#     for spine in ax.spines.values():
#         spine.set_edgecolor('#d3d3d3')

#     # ax.xaxis.grid(color='#a9a9a9', linestyle='dashed', linewidth=0.5)
#     # ax.yaxis.grid(color='#a9a9a9', linestyle='dashed', linewidth=0.5)
#     #plt.grid('minor')

#     ax.set_title('Stacked Bar Chart (Percentage)', color='#a9a9a9')
#     ax.set_xlabel('')
#     ax.set_ylabel('Percentage', color='#a9a9a9')

#     # Set font color for legend
#     legend = ax.legend(title='Y')
#     legend.get_title().set_color('#a9a9a9')

#     # Save figure to a buffer
#     buffer_stacked_bar_percentage = BytesIO()
#     fig.savefig(buffer_stacked_bar_percentage, format='png')
#     image_stacked_bar_percentage = base64.b64encode(buffer_stacked_bar_percentage.getvalue()).decode('utf-8')
#     plt.close()

#     return image_stacked_bar_percentage


def nested_bar_chart(x, y, width, height):
    # Create a DataFrame from the categorical variables
    df = pd.DataFrame({'X': x, 'Y': y})

    # Create a count plot
    fig, ax = plt.subplots(figsize=(width, height))
    sns.countplot(x='X', hue='Y', data=df, palette='Set3', ax=ax)


    ax.set_title('Nested Bar Chart', color='#a9a9a9')
    ax.set_xlabel(x.name, color='#a9a9a9')
    ax.set_ylabel('Count', color='#a9a9a9')

    # Set font color for legend
    legend = ax.legend(title=y.name)
    legend.get_title().set_color('#a9a9a9')

    # Crop x-axis ticks if longer than 20 symbols
    labels = [text.get_text() if len(text.get_text()) <= 10 else text.get_text()[:10] + '...' for text in ax.get_xticklabels()]
    ax.set_xticks(ax.get_xticks())  # Set the tick positions
    ax.set_xticklabels(labels, ha='center')

    # Set font color for all text in the chart
    for text in ax.texts + ax.get_xticklabels() + ax.get_yticklabels():
        text.set_color('#a9a9a9')

    # Set frame and grid color to a9a9a9
    for spine in ax.spines.values():
        spine.set_edgecolor('#d3d3d3')
    #ax.grid(color='#e6e6e6')

    # Save figure to a buffer
    buffer_nested_bar = BytesIO()
    fig.savefig(buffer_nested_bar, format='png')
    image_nested_bar = base64.b64encode(buffer_nested_bar.getvalue()).decode('utf-8')
    plt.close()

    return image_nested_bar

# def stacked_bar_chart_percentage(x, y,*k):
#     # Create a DataFrame from the categorical variables
#     df = pd.DataFrame({'X': x, 'Y': y})

#     # Create a cross-tabulation
#     cross_tab = pd.crosstab(df['X'], df['Y'])

#     # Normalize the data to get percentages
#     cross_tab_percentage = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100

#     # Prettify the style
#     sns.set(style="whitegrid", rc={"grid.color": "#f0f0f0", "axes.edgecolor": "#f0f0f0", "axes.linewidth": 1.0})
#     plt.figure(figsize=(8, 6))

#     # Plotting the stacked bar chart with percentages
#     colors = sns.color_palette("pastel")  # You can change the palette as needed
#     ax = cross_tab_percentage.plot(kind='bar', stacked=True, color=colors, edgecolor='none')

#     # Rotate x-axis tick labels by 90 degrees and crop if longer than 20 symbols
#     labels = [str(cls) for cls in cross_tab_percentage.index]
#     ax.set_xticklabels(labels, rotation=0, ha='center')

#     # Set the text color for ylabel, legend, and title to light grey
#     ax.yaxis.label.set_color('#a9a9a9')
#     ax.legend(title='Y', loc='upper right').get_title().set_color('#a9a9a9')
#     ax.set_title('Stacked Bar Chart (Percentage)', color='#a9a9a9')

#     # Set the text color for x-axis and y-axis tick labels to light grey
#     ax.tick_params(axis='x', colors='#a9a9a9')
#     ax.tick_params(axis='y', colors='#a9a9a9')

#     plt.xlabel(None)
#     plt.ylabel('Percentage', color='#a9a9a9')
#     #plt.grid(False)

#     # Save figure to a buffer
#     buffer_stacked_bar_percentage = BytesIO()
#     plt.savefig(buffer_stacked_bar_percentage, format='png')
#     image_stacked_bar_percentage = base64.b64encode(buffer_stacked_bar_percentage.getvalue()).decode('utf-8')
#     plt.close()

#     return image_stacked_bar_percentage




# def nested_bar_chart(x, y,*k):
#     # Create a DataFrame from the categorical variables
#     df = pd.DataFrame({'X': x, 'Y': y})

#     # Prettify the style
#     sns.set(style="whitegrid", rc={"grid.color": "#f0f0f0", "axes.edgecolor": "#f0f0f0", "axes.linewidth": 1.0})
#     plt.figure(figsize=(8, 6))

#     # Create a count plot
#     sns.countplot(x='X', hue='Y', data=df, palette='Set3', edgecolor='none')

#     plt.title('Nested Bar Chart', color='#a9a9a9')
#     plt.xlabel(None)
#     plt.ylabel('Count', color='#a9a9a9')
    
#     # Get legend title and set its color
#     legend = plt.legend(title='Y')
#     legend.get_title().set_color('#a9a9a9')

#     # Set the legend text color to light grey
#     for text in legend.get_texts():
#         text.set_color('#a9a9a9')

#     # Set the text color for x-axis tick labels to light grey
#     plt.xticks(color='#a9a9a9')
#     plt.yticks(color='#a9a9a9')

#     # Save figure to a buffer
#     buffer_nested_bar = BytesIO()
#     plt.savefig(buffer_nested_bar, format='png')
#     image_nested_bar = base64.b64encode(buffer_nested_bar.getvalue()).decode('utf-8')
#     plt.close()

#     return image_nested_bar


#abab


###########################################################################################################################
######################################### TEXT CHARTS ###########################################################
###########################################################################################################################



def word_cloud(data, cfg, name=None):
    # Combine all text in the specified column
    text = ' '.join(data.astype(str))
    
    cfg = {
    'width': 300,
    'height': 200,
    'fig_width': 5,
    'fig_height': 3.334,
    'max_words': 50,
    }

    # Generate the word cloud
    try:
        wordcloud = WordCloud(width=cfg['width'], height=cfg['height'], background_color='white',max_words=cfg['max_words']).generate(text)
    except ValueError:
        wordcloud = WordCloud(width=cfg['width'], height=cfg['height'], background_color='white',max_words=cfg['max_words']).generate('untokenable')
    # Access the tokenized words using the process_text method
    #tokens = wordcloud.process_text(text)
    tokens = word_tokenize(text)

    # Plot the WordCloud image
    fig_wordcloud, ax_wordcloud = plt.subplots(figsize=(cfg['fig_width'], cfg['fig_height']))
    ax_wordcloud.imshow(wordcloud, interpolation='bilinear')
    ax_wordcloud.axis('off')  # Turn off axis labels
    if name is not None:
        ax_wordcloud.set_title(name)
    ax_wordcloud.set_frame_on(False)
    plt.grid(False)

    # Save the figure to a buffer
    buffer_wordcloud = BytesIO()
    fig_wordcloud.savefig(buffer_wordcloud, format='png')
    image_wordcloud_thumb = base64.b64encode(buffer_wordcloud.getvalue()).decode('utf-8')
    plt.close(fig_wordcloud)

    return image_wordcloud_thumb, tokens

# Example usage
# Assuming you have a DataFrame named 'df' with a column 'text_column'






###########################################################################################################################
######################################### CATEGORICAL CHARTS ###########################################################
###########################################################################################################################


def bar_chart_thumb(data, cfg, threshold=0.15):
    """
    Generate a horizontal bar chart.

    Parameters:
    - data (pandas.Series or pandas.DataFrame): Data for the bar chart.
    - threshold (float): Proportion threshold to decide whether to place the count inside or outside the bar.

    Returns:
    - str: Base64-encoded PNG image of the bar chart.
    """
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()  # Convert DataFrame to Series if needed

    value_counts = data.value_counts()

    # Check if there are more than 5 unique values
    if len(value_counts) > 5:
        # Combine smaller categories into 'others'
        others_count = value_counts.iloc[5:].sum()
        value_counts = pd.concat([value_counts.head(5), pd.Series([others_count], index=['others'])])

    total_count = value_counts.sum()
    

    # Calculate a proportional width based on the number of bars
    max_width = 0.8
    width = max_width if len(value_counts) > 2 else 0.4
    length_factor = 1.5  # Adjust this factor to control the influence of name length

    #'bar_color': 'light:#474B7B'
    col_in = cfg['bar_color']
    colors = sns.color_palette(col_in)
    
    fig_bar, ax_bar = plt.subplots(figsize=(8, 3))  # Customizable width and height
    bars = value_counts.plot(kind='barh', color=colors, edgecolor='none', ax=ax_bar, width=width)  # Set edgecolor to 'none' for no edges

    ax_bar.xaxis.set_visible(False)

    ax_bar.set_frame_on(False)
    plt.grid(False)

    # Hide y-axis label
    ax_bar.set_ylabel('')

    # Increase the size of the y-axis ticks
    ax_bar.tick_params(axis='y', which='major', labelsize=12)  # Adjust the labelsize as needed

    # Add names at the center or outside of each bar with a larger font size
    for bar, name in zip(bars.patches, value_counts.index):
        width, height = bar.get_width(), bar.get_height()
        count = int(width)
        proportion = width / total_count
        length_factor = (len(str(name))+2) * 0.045  # Adjust this factor to control the influence of name length

        
        if proportion / length_factor < threshold:
            ax_bar.text(width + 0.01, bar.get_y() + height/2, f' {name}', va='center', ha='left', color='#636363', size=14)
        else:
            ax_bar.text(width/2, bar.get_y() + height/2, f'{name}', va='center', ha='center', color=cfg['bar_font_color'], size=14)

    # Set a maximum of 6 ticks
    num_ticks = min(len(value_counts), 6)
    
    # Set y-axis ticks and labels
    ax_bar.set_yticks(range(num_ticks))
    # Assuming your keys are the same as the range of the length of the array
    ax_bar.set_yticklabels(value_counts.values)


    ax_bar.yaxis.set_visible(True)
    #print(value_counts.values)
    # Save figure to a buffer
    buffer_bar = BytesIO()
    fig_bar.savefig(buffer_bar, format='png')
    image_bar_thumb = base64.b64encode(buffer_bar.getvalue()).decode('utf-8')
    plt.close(fig_bar)

    return image_bar_thumb


def pie_chart(data_series, cfg, name=None, min_percentage=5, max_categories=5, explode_factor=0.05):
    # Use value_counts to get counts and labels
    value_counts = data_series.value_counts()

    # Extract data and labels from the value_counts result
    data = value_counts.values
    labels = value_counts.index

    # Calculate total sum
    total_sum = sum(data)

    # Calculate proportions
    proportions = data / total_sum

    # Find indices of categories exceeding the minimum percentage
    filtered_indices = np.where(proportions >= min_percentage / 100)[0]

    # Sort categories by proportion and limit to maximum categories
    sorted_indices = np.argsort(proportions[filtered_indices])[::-1][:max_categories]

    # Extract data and labels of filtered and sorted categories
    data = data[filtered_indices][sorted_indices]
    labels = labels[filtered_indices][sorted_indices]

    # Calculate total sum of included categories
    included_sum = np.sum(data)

    # Calculate proportion of "Others" category
    others_sum = total_sum - included_sum

    # Combine categories that are below the threshold into "Others"
    if others_sum > 0:
        data = np.append(data, others_sum)
        labels = np.append(labels, 'Others')

    # Calculate explode values
    num_categories = len(data)
    explode = [explode_factor] * num_categories

    # Create a pie chart
    fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
    ax_pie.pie(data, labels=labels, autopct='%1.1f%%', colors=sns.color_palette('pastel'), startangle=50, explode=explode)

    ax_pie.tick_params(axis='both', colors='#a9a9a9')
    if name is not None:
        ax_pie.set_title(name)
    ax_pie.set_frame_on(False)
    plt.grid(False)

    # Save the figure to a buffer
    buffer_pie = BytesIO()
    fig_pie.savefig(buffer_pie, format='png')
    image_pie_thumb = base64.b64encode(buffer_pie.getvalue()).decode('utf-8')
    plt.close(fig_pie)

    return image_pie_thumb

def Nightndale_chart(data_series, cfg, name=None):
    #Use value_counts to get counts and labels
    value_counts = data_series.value_counts()

    # Extract data and labels from the value_counts result
    data = value_counts.values
    labels = value_counts.index

    # Limit to the top 5 categories
    top_categories = 10
    if len(labels) > top_categories:
        data = data[:top_categories]
        labels = labels[:top_categories]
        # Combine the remaining categories into "Others"
        data = np.append(data, sum(value_counts.values[top_categories:]))
        labels = np.append(labels, 'Others')

    # Convert data to radians
    theta = np.linspace(0.0, 2 * np.pi, len(labels), endpoint=False)

    # Create a Nightingale diagram
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 4))
    ax.fill(theta, data, color='skyblue', alpha=0.75)

    # Customize the plot
    ax.set_theta_offset(np.pi / 2)  # Adjust the starting angle if needed
    ax.set_theta_direction(-1)  # Change to 1 for clockwise direction
    ax.set_rlabel_position(0)  # Move radial labels to the center
    ax.set_yticklabels([])  # Hide radial labels if not needed

    # Add category labels
    ax.set_xticks(theta)
    ax.set_xticklabels(labels)

    if name is not None:
        ax.set_title(name)

    # Save the figure to a buffer
    buffer_nightingale = BytesIO()
    fig.savefig(buffer_nightingale, format='png')
    image_nightingale_thumb = base64.b64encode(buffer_nightingale.getvalue()).decode('utf-8')
    plt.close(fig)

    return image_nightingale_thumb



def radial_bar_chart(data_series,cfg):
    # Extract up to 5 categories and values from the data series
    value_counts = data_series.value_counts()

    # Extract data and labels from the value_counts result
    data = value_counts.values
    labels = value_counts.index

    # Limit to the top categories
    top_categories = 10
    if len(labels) > top_categories:
        data = data[:top_categories]
        labels = labels[:top_categories]
        # Combine the remaining categories into "Others"
        data = np.append(data, sum(value_counts.values[top_categories:]))
        labels = np.append(labels, 'Others')

    theta = np.linspace(0.0, 2 * np.pi, len(labels), endpoint=False)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.bar(theta, data, color=['skyblue', 'orange', 'lightgreen', 'pink', 'lightcoral'], alpha=0.7)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)  # Move radial labels to the center
    ax.set_yticklabels([])  # Hide radial labels if not needed

    ax.set_xticks(theta)
    ax.set_xticklabels(labels)

    plt.title('Radial Bar Chart')

    # Save figure to a buffer
    buffer_radial = BytesIO()
    plt.savefig(buffer_radial, format='png')
    image_radial = base64.b64encode(buffer_radial.getvalue()).decode('utf-8')
    plt.close(fig)

    return image_radial




def treemap(data_series, cfg):
    # Use value_counts to get counts and labels
    value_counts = data_series.value_counts()

    # Extract data and labels from the value_counts result
    data = value_counts.values
    labels = value_counts.index
    top_categories=5
    # Limit to the top categories
    if len(labels) > top_categories:
        data = data[:top_categories]
        labels = labels[:top_categories]
        # Combine the remaining categories into "Others"
        data = np.append(data, sum(value_counts.values[top_categories:]))
        labels = np.append(labels, 'Others')

    squarify.plot(sizes=data, label=labels, color=['skyblue', 'orange', 'lightgreen', 'pink', 'lightcoral'], alpha=0.7)
    plt.axis('off')
    plt.title('Treemap')

    # Save figure to a buffer
    buffer_treemap = BytesIO()
    plt.savefig(buffer_treemap, format='png')
    image_treemap = base64.b64encode(buffer_treemap.getvalue()).decode('utf-8')
    plt.close()

    return image_treemap



def donut_chart(data_series, cfg):
    # Use value_counts to get counts and labels
    value_counts = data_series.value_counts()

    # Extract data and labels from the value_counts result
    data = value_counts.values
    labels = value_counts.index
    top_categories=5
    # Limit to the top categories
    if len(labels) > top_categories:
        data = data[:top_categories]
        labels = labels[:top_categories]
        # Combine the remaining categories into "Others"
        data = np.append(data, sum(value_counts.values[top_categories:]))
        labels = np.append(labels, 'Others')

    plt.pie(data, labels=labels, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4), colors=['skyblue', 'orange', 'lightgreen', 'pink', 'lightcoral'])
    plt.axis('equal')  # Equal aspect ratio ensures the donut is circular.
    plt.title('Donut Chart')

    # Save figure to a buffer
    buffer_donut = BytesIO()
    plt.savefig(buffer_donut, format='png')
    image_donut = base64.b64encode(buffer_donut.getvalue()).decode('utf-8')
    plt.close()

    return image_donut

##################################################################################################################
def hist_plot_thumb(data,cfg, name=None):
    # Create a histogram
    fig_histogram, ax_histogram = plt.subplots(figsize=(6, 4))
    ax_histogram.hist(data, bins=30, color=cfg['hist_color'], edgecolor='white', density=True) #'skyblue'
    
    # Add a kernel density plot
    sns.kdeplot(data, color=cfg['hist_kde_color'], ax=ax_histogram, warn_singular=False) #'pink'
    
    ax_histogram.tick_params(axis='both', colors='#a9a9a9')
    ax_histogram.set_xlabel('Value', color='#a9a9a9')
    ax_histogram.set_ylabel('Frequency', color='#a9a9a9')
    if name is not None:
        ax_histogram.set_title(name)
    ax_histogram.set_frame_on(False)
    plt.grid(False)

    # Save the figure to a buffer
    buffer_histogram = BytesIO()
    fig_histogram.savefig(buffer_histogram, format='png')
    image_histogram_thumb = base64.b64encode(buffer_histogram.getvalue()).decode('utf-8')
    plt.close(fig_histogram)

    return image_histogram_thumb


###########################################################################################################################
######################################### WRAPPER ###########################################################
###########################################################################################################################




def missing_plot(data, cfg, name=None):
    # Calculate the percentage of missing values for each column
    missing_percentage = data.isnull().mean() * 100
    
    # Create a bar plot with missing values on the upper part and rotated bottom ticks
    fig, ax = plt.subplots(figsize=(12, 4))  # Adjusted the width to provide more space
    
    # Set the bar width (adjust this value as needed)
    bar_width = 0.5
    
    # Plot blue bars for non-missing values
    ax.bar(missing_percentage.index, 100 - missing_percentage, color=cfg['hist_color'], label='All Values', width=bar_width)
    
    # Plot red bars for missing values on the upper part
    ax.bar(missing_percentage.index, missing_percentage, bottom=100 - missing_percentage, color=cfg['hist_kde_color'], label='Missing Values', width=bar_width)
    
    # Customize the plot
    ax.set_ylabel('Percentage')
    ax.set_title('Missing Values Bar Chart',pad=20)
    ax.legend()
    
    # Rotate the bottom ticks for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout to prevent x-axis tick names from being cut out
    plt.tight_layout()
    
    # Remove the frame and grid
    ax.set_frame_on(False)
    plt.grid(False)

    # Save the figure to a buffer
    buffer_plot = BytesIO()
    fig.savefig(buffer_plot, format='png')
    missing_plot = base64.b64encode(buffer_plot.getvalue()).decode('utf-8')
    plt.close(fig)

    return missing_plot


###########################################################################################################################
######################################### DASH ###########################################################
###########################################################################################################################


def triple_donut(data, cfg, name=None):
    #print(data)
    # Data for the three donut charts
    data1 = data['Missing cells (%)']  # Replace with your desired percentage
    data2 = data['Duplicate rows (%)']  # Replace with your desired percentage
    data3 = data['Constant columns (%)'] # Replace with your desired percentage
    
    # Colors for the donut charts
    color1 = '#DCDCDC' # Background color
    color2 = cfg['dash_donuts_color']      # Color corresponding to the value
    
    # Create subplots with three donut charts
    fig, ax = plt.subplots(1, 3, figsize=(6, 1.75))
    
    # Function to draw a donut chart with a percentage value in the center
    def draw_donut(ax, data, title):
        ax.pie([100 - data, data], colors=[color1, color2], autopct='', startangle=90,
               wedgeprops=dict(width=0.3, edgecolor='w'))  # Add wedgeprops to create a hole in the center
    
        # Add the percentage value in the center
        ax.text(0, 0, f'{data}%', ha='center', va='center', fontsize=16, weight='bold', color=color2)
    
        ax.set_title(title)
    
    # Donut chart 1
    draw_donut(ax[0], data1, 'Missing cells')
    
    # Donut chart 2
    draw_donut(ax[1], data2, 'Duplicate rows')
    
    # Donut chart 3
    draw_donut(ax[2], data3, 'Constant columns')
    
    # Equal aspect ratio ensures that pie is drawn as a circle.
    for a in ax:
        a.axis('equal')
    
    # Save the figure to a buffer
    buffer_image = BytesIO()
    fig.savefig(buffer_image, format='png')
    image_thumb = base64.b64encode(buffer_image.getvalue()).decode('utf-8')
    plt.close(fig)


    return image_thumb



def triple_box(data, cfg, name=None):
    # Create a histogram
    fig, ax = plt.subplots(figsize=(6, 2))

    values = {' Categorical':data['Categorical'],
            ' Numeric':data['Numeric'],
            ' Text':data['Text'],
            ' Datetime':data['Datetime'],
    }
    # Assuming 'data' is a list of four values

    # Filter out values that are <= 0
    filtered_values = [value for k,value in values.items() if value > 0]
    max_value = max(filtered_values)
    
    # Custom bar labels (you can modify these as needed)
    bar_labels = [k for k,value in values.items() if value > 0]
    
    # Define a color palette for bars
    col_in = "light:" + cfg['dash_bars_color']
    colors = sns.color_palette(col_in,len(filtered_values))
    
    bars = ax.barh(bar_labels, filtered_values, color=colors, edgecolor='white', alpha=0.7)

    # Add labels inside the bars for values > 0
    for bar, value, label in zip(bars, filtered_values, bar_labels):
        if value > 0:
            text = f'{label}: {value}'
            # If the value is 1, move the label to the right side
            if (value / max_value ) < 0.16:
                ax.text(value + 0.2, bar.get_y() + bar.get_height() / 2, text, va='center', ha='left', color=cfg['dash_bars_text_color'])
            else:
                ax.text(value / 2, bar.get_y() + bar.get_height() / 2, text, va='center', ha='center', color=cfg['dash_bars_text_color'])

    ax.tick_params(axis='both', colors='#a9a9a9')
    ax.set_xlabel('Values', color='#a9a9a9')
    ax.set_ylabel('Category', color='#a9a9a9')

    # Set y-axis ticks and labels
    ax.set_yticks(range(len(filtered_values)))
    ax.set_yticklabels(filtered_values)

    if name is not None:
        ax.set_title(name)

    ax.set_frame_on(False)
    plt.grid(False)

    # Save the figure to a buffer
    buffer_img = BytesIO()
    fig.savefig(buffer_img, format='png')
    image_thumb = base64.b64encode(buffer_img.getvalue()).decode('utf-8')
    plt.close(fig)

    return image_thumb

###########################################################################################################################
######################################### CORRELATIONS ###########################################################
###########################################################################################################################

#wiz_corr

def correlation_plot(corr_matrix):
    # Create a heatmap
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(corr_matrix['data'], annot=True, cmap='YlGnBu', fmt='.2f')

    # Rotate x-axis labels by 45 degrees
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45)

    # Truncate x-axis tick labels if their length exceeds 8 characters
    xticks = heatmap.get_xticklabels()
    xtick_labels = [label.get_text() for label in xticks]  # Extract text from Text objects
    xtick_labels = [label[:7] + '.' if len(label) > 7 else label for label in xtick_labels]
    heatmap.set_xticklabels(xtick_labels)

    # Truncate y-axis tick labels if their length exceeds 12 characters
    yticks = heatmap.get_yticklabels()
    ytick_labels = [label.get_text() for label in yticks]  # Extract text from Text objects
    ytick_labels = [label[:11] + '.' if len(label) > 11 else label for label in ytick_labels]
    heatmap.set_yticklabels(ytick_labels)

    # Set font color for axis labels and title
    plt.title('Correlation Plot', color='#a9a9a9')
#    plt.xlabel('Variables', color='#a9a9a9')
#    plt.ylabel('Variables', color='#a9a9a9')

    # Set font color for ticks
    plt.xticks(color='#a9a9a9')
    plt.yticks(color='#a9a9a9')

    # Save figure to a buffer
    buffer_correlation = BytesIO()
    plt.savefig(buffer_correlation, format='png')
    image_correlation = base64.b64encode(buffer_correlation.getvalue()).decode('utf-8')
    plt.close()

    return image_correlation
    
###########################################################################################################################
######################################### WIZ ###########################################################
###########################################################################################################################

def wiz_dash(hub,cfg):
    """
    Create visualizations for plot(all_df)
    """
    
    dfw, data = hub["col"], hub["data"]

    # im passing the v_type from comp already, here inseted of continuous, it changes is to numerical
    #v_type = 'numerical'
    #data['v_type'] = v_type
    
    #pic_thumb = hist_plot_thumb(col,cfg)
    pic_thumb = missing_plot(dfw,cfg)
    #toto = tab_info(hub,cfg)

    donut = triple_donut(data,cfg)
    dash_b = triple_box(data,cfg)
    #print('df')

    
    
    # Store the results in variables to avoid repeated dictionary lookups
    stat_tab = generate_table(data, container_keysW, 'val_tableW',style, pic_thumb, cfg=cfg)
    val_tabx = value_tableW(dfw,cfg)
    #qq = qq_plot(col,cfg)

    plots = {
        'type': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm'],
        'title': ['Dash','Stats', 'Missing', 'Value Table'],
        'image': [((donut,500),(dash_b,500),),None, ((pic_thumb,1000),), None],
        'image_width': 1000,
        'value_table': [None,stat_tab, None, val_tabx,None],
        
    }
    
    return plots

def wiz_corr(hub, cfg):
    
    """
    Create visualizations for correlations(df)
    """

    #print(hub)
    store = {
        # 'val_tab2': value_tableH(data,pic),
        # 'val_tab':value_tableH(data,pic2,pic3),
        'cr'  : correlation_plot(hub),
        # 'bx': box_plot_con(col1,col2),
        # 'lc' : line_chart(col1,col2),
        # 'box' : box_plot(col),
    }

    plots = {
            'type': ['a','b','c','d','e','f','g','h','i','j','k','l','m'
            ],
        
            'title': [#'iq',
                      #'Value Table',
                      'Corr',
                      # 'Box',
                      # 'Line',
                      #'Box Plot'
                    ],
        
            'image': [#None,
                      #None,
                      ((store['cr'],800),),
                      # ((store['bx'],800),),
                      # ((store['lc'],800),),
                      #store['box']
                    ],
        
            'value_table': [#store['val_tab2'],
                            #store['val_tab'],
                            None,
                            # None,
                            # None,
                            #None
            ],
        }
    
    return plots

def wiz_biv_cat(hub):
    """
    Create visualizations for plot(df, Continuous)
    """

    col1,col2, data = hub["col1"], hub["col2"], hub["data"]
    #print(col,data)
    #tabs: List[Panel] = []
    # pic = hist_plot(col)
    # pic2 = qq_plot(col)
    # pic3 = kde_plot(col)
    #htgs: Dict[str, List[Tuple[str, str]]] = {}

    # Set the width and height for both charts
    width, height = 8, 6



#aaa


    store = {
        # 'val_tab2': value_tableH(data,pic),
        # 'val_tab':value_tableH(data,pic2,pic3),
        'hm': heatmap_plot(col1,col2),
        'nbch' : nested_bar_chart(col1,col2,width, height),
        'sbch'  : stacked_bar_chart_percentage(col1,col2,width, height),
        # 'box' : box_plot(col),
    }

    plots = {
            'type': ['a','b','c','d','e','f','g','h','i','j','k','l','m'
            ],
        
            'title': [#'iq',
                      #'Value Table',
                      'HM',
                      'Nest',
                      'Stacked',
                      #'Box Plot'
                    ],
        
            'image': [#None,
                      #None,
                      ((store['hm'],800),), #800 for all
                      ((store['nbch'],800),),
                      ((store['sbch'],800),),
                      #store['box']
                    ],
        
            'value_table': [#store['val_tab2'],
                            #store['val_tab'],
                            None,
                            None,
                            None,
                            #None
            ],
        }
    
    return plots
def wiz_biv_con_cat(hub):
    """
    Create visualizations for plot(df, Continuous)
    """
    col1,col2, data = hub["col1"], hub["col2"], hub["data"]
    #print(col1.name)

    #(('categorical', 2, 0.002244668911335578), ('continuous', 88, 0.09876543209876543))
    if data[0][0] == 'continuous':
        col1,col2 = col2,col1
    #print(data)
    
    #print(col,data)
    #tabs: List[Panel] = []
    # pic = hist_plot(col)
    # pic2 = qq_plot(col)
    # pic3 = kde_plot(col)
    #htgs: Dict[str, List[Tuple[str, str]]] = {}






    store = {
        # 'val_tab2': value_tableH(data,pic),
        # 'val_tab':value_tableH(data,pic2,pic3),
        'vp'  : violin_plot(col1,col2),
        'bx': box_plot_con(col1,col2),
        #'lc' : line_chart(col1,col2),
        # 'box' : box_plot(col),
    }

    plots = {
            'type': ['a','b','c','d','e','f','g','h','i','j','k','l','m'
            ],
        
            'title': [#'iq',
                      #'Value Table',
                      'Violin',
                      'Box',
                      #'Line',
                      #'Box Plot'
                    ],
        
            'image': [#None,
                      #None,
                      ((store['vp'],800),),
                      ((store['bx'],800),),
                      #((store['lc'],800),),
                      #store['box']
                    ],
        
            'value_table': [#store['val_tab2'],
                            #store['val_tab'],
                            None,
                            None,
                            #None,
                            #None
            ],
        }
    
    return plots
#axxx
def wiz_biv_con(hub):
    """
    Create visualizations for plot(df, Continuous)
    """
    col1,col2, data = hub["col1"], hub["col2"], hub["data"]
    #print(col,data)
    #tabs: List[Panel] = []
    # pic = hist_plot(col)
    # pic2 = qq_plot(col)
    # pic3 = kde_plot(col)
    #htgs: Dict[str, List[Tuple[str, str]]] = {}






    store = {
        # 'val_tab2': value_tableH(data,pic),
        # 'val_tab':value_tableH(data,pic2,pic3),
        'sc': scatter_plot(col1,col2),
        'hx' : hexbin_plot(col1,col2),
        # 'qq'  : qq_plot(col),
        # 'box' : box_plot(col),
    }

    plots = {
            'type': ['a','b','c','d','e','f','g','h','i','j','k','l','m'
                     
            ],
        
            'title': [#'iq',
                      #'Value Table',
                      'Scatter',
                      'Hexbin',
                      #'QQ Plot',
                      #'Box Plot'
                    ],
        
            'image': [#None,
                      #None,
                      ((store['sc'],800),),
                      ((store['hx'],800),),
                      #store['qq'],
                      #store['box']
                    ],
        
            'value_table': [#store['val_tab2'],
                            #store['val_tab'],
                            None,
                            None,
                            #None,
                            #None
            ],
        }
    
    return plots

container_keysN = {
        'container1': ['VALUES', 'MISSING', 'DISTINCT', '', 'MEMORY', 'DTYPE'],
        'container2': ['MAX', '95%', 'Q3', 'AVG', 'MEDIAN', 'Q1', '5%', 'MIN'],
        'container3': ['RANGE', 'IQR', 'STD', 'VAR', ' ', 'KURT.', 'SKEW', 'SUM']
    }

container_keysT = {
        'container1': ['VALUES', 'MISSING', 'DISTINCT', '', 'MEMORY', 'DTYPE'],
        'container2': ['Max length', 'Mean length','Median length', 'Min length'],
        'container3': ['RANGE', 'IQR', 'STD', 'VAR', ' ', 'KURT.', 'SKEW', 'SUM']
    }

container_keysC = {
        'container1': ['VALUES', 'MISSING', 'DISTINCT', '', 'MEMORY', 'DTYPE'],
        'container3a': [0,1,2,3,4,5]
        #'container3': ['RANGE', 'IQR', 'STD', 'VAR', ' ', 'KURT.', 'SKEW', 'SUM']
    }

container_keysW = {
        'Stats': ['Number of columns', 'Number of rows', 'Missing cells', 'Missing cells (%)', 'Duplicate rows', 'Duplicate rows (%)','Total memory size in KB','Avg. record size in Bytes'],
        'Columns': ['Numeric', 'Categorical', 'Text', 'Datetime',],
        'Alerts': ['missing', 'unique', 'constant', 'zero', 'negative'],
    }



def wiz_text(hub,cfg):

    """
    Create visualizations for plot(df, Text)
    """

    col, data = hub["col"], hub["data"]

    w_len = data['w_len']
    #pic = hist_plot_thumb(w_len,cfg,name="Value Lengths")
    
    pic, tokens = word_cloud(col, cfg)

    


    from collections import Counter
    
    # Basic Analysis: Calculate word frequencies
    #word_freq = Counter(tokens)
    #most_common_words = word_freq.most_common(10)
    #print("Most common words:", most_common_words)
    serial_tokens = pd.Series(tokens).sort_values(ascending=False)

    
    # print(len(tokens))
    # print(serial_tokens.value_counts())
    # print(col)
    #print(ser_tokens)

    
    store = {
        'stat_tab': generate_table(data,container_keysT,'val_tableT', style, pic,cfg=cfg),
        #'val_tab1': value_tableT(data,style,pic),
        'val_tabx':value_tableX(col,cfg,tokens=serial_tokens),
        
        #'hist': hist_plot(col),
        #'kde' : kde_plot(col),
        #'qq'  : qq_plot(col),
        #'box' : box_plot(col),
    }
    toto = tab_info(hub,cfg)

    plot_info = {
        'type': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm'],
        'title': [toto, 'Stats', 'Value Table'],
        'image': [None, None, None],
        'value_table': [None, store['stat_tab'], store['val_tabx']],
    }

    return plot_info


def wiz_cat(hub,cfg):
    """
    Create visualizations for plot(df, Categorical)
    """
    col, data = hub["col"], hub["data"]
    
    # Pass the v_type from comp already; here instead of categorical, it changes to categorical
    # v_type = 'categorical'
    # data['v_type'] = v_type

    pic = bar_chart_thumb(col,cfg)
    
    pie = pie_chart(col,cfg)
    #rad = radial_bar_chart(col,cfg)

    # fake_dict = {1:'a',2:'b',3:'c',4:'d',5:'e'}
    # data.update(fake_dict)
    
    toto = tab_info(hub,cfg)
    
    # Store the results in variables to avoid repeated dictionary lookups
    stat_tab = generate_table(data, container_keysC, 'val_tableC', style, pic,cfg=cfg)
    val_tabx = value_tableX(col,cfg)
    #hist = hist_plot(col,cfg)

    plots = {
        'type': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm'],
        'title': [toto, 'Stats', 'Pie Chart', 'Value Table'],
        'image': [None, None, ((pie,500),), None],
        'value_table': [None, stat_tab, None, val_tabx],
    }
    
    return plots

def wiz_num(hub,cfg):
    """
    Create visualizations for plot(df, Continuous)
    """
    
    col, data = hub["col"], hub["data"]

    # im passing the v_type from comp already, here inseted of continuous, it changes is to numerical
    #v_type = 'numerical'
    #data['v_type'] = v_type
    
    pic_thumb = hist_plot_thumb(col,cfg)
    
    toto = tab_info(hub,cfg)
    
    # Store the results in variables to avoid repeated dictionary lookups
    stat_tab = generate_table(data, container_keysN, 'val_tableH', style, pic_thumb,cfg=cfg)
    val_tabx = value_tableX(col,cfg)
    qq = qq_plot(col,cfg)

    plots = {
        'type': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm'],
        'title': [toto, 'Stats', 'QQ Plot', 'Value Table'],
        'image': [None, None, ((qq,500),), None],
        'value_table': [None, stat_tab, None, val_tabx],
    }
    
    return plots

def wiz_dt(hub,cfg):

    """
    Create visualizations for plot(df, dt)
    """
    col, data = hub["col"], hub["data"]
    
    # Pass the v_type from comp already; here instead of categorical, it changes to categorical
    # v_type = 'categorical'
    # data['v_type'] = v_type

    pic = hist_plot_thumb(col,cfg)
    #pic = bar_chart_thumb(col,cfg)
    #pie = pie_chart(col,cfg)
    
    toto = tab_info(hub,cfg)
    
    # Store the results in variables to avoid repeated dictionary lookups
    stat_tab = generate_table(data, container_keysN, 'val_tableH', style, pic,cfg=cfg)
    val_tabx = value_tableX(col,cfg)
    #hist = hist_plot(col,cfg)

    plots = {
        'type': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm'],
        'title': [toto, 'Stats', 'Value Table'],
        'image': [None, None, None],
        'value_table': [None, stat_tab, val_tabx],
    }
    
    return plots

def tab_info(hub,cfg):

    #print(hub)
    name = hub["col"].name
    col, data = hub["col"], hub["data"]
    

    """" just copy of render num """
    pic_mini = hist_plot_thumb_mini(col,cfg)
    
    if data['MISSING_P'] > 0:
        font_size = '13px'
        font_color = '#FF6961'
        font_weight = '500'
    else:
        font_size = '11px'
        font_color = '#a9a9a9'
        font_weight = '300'
        
    if data['DISTINCT'] == 1 or data['DISTINCT_P'] == 100:
        font_sizeD = '13px'
        font_colorD = '#FF6961' #red
        font_weightD = '500'
    else:
        font_sizeD = '11px'
        font_colorD = '#a9a9a9'
        font_weightD = '300'


    return f"""
        <div class="top_tab" style="width: 750px; height: 22px; padding: 0px 10px 0px; background-color: white; display: flex; align-items: center;">
            <div style="padding: 12px 10px 0px; width: 270px;">
                    <div style="font-size: 18px; height:20px;">{name}</div>
                    <div class="bottom_line" style="font-size: 12px; color: #a9a9a9; font-weight: 500; height:20px;">{data['v_type']}</div>
            </div>
                           <div class="table_tab" style="width: 180px; font-size: 10px; margin-bottom: 0px;">
                                <div style="display: flex;">
                        
                                    <div style="width: 60px; font-size: 11px; text-align: center; padding: 10px 5px; color: #a9a9a9; font-weight: 300;"> {data['VALUES_P']}%</div>
                                     <div style="width: 60px; font-size: {font_size}; text-align: center; padding: 10px 5px; color: {font_color}; font-weight: {font_weight};"> {data['MISSING_P']}%</div>
                                    <div style="width: 60px; font-size: {font_sizeD}; text-align: center; padding: 10px 5px; color: {font_colorD}; font-weight: {font_weightD};"> {data['DISTINCT_P']}%</div>
                                   
                                </div>
                                <div style="display: flex;">
                        
                                    
                                    <div style="width: 60px; font-size: 10px; text-align: center; padding: 0px 5px; color: #a9a9a9; font-weight: 300;"> VALID</div>
                                    <div style="width: 60px; font-size: 10px; text-align: center; padding: 0px 5px; color: #a9a9a9; font-weight: 300;"> MISSING</div>
                                    <div style="width: 60px; font-size: 10px; text-align: center; padding: 0px 5px; color: #a9a9a9; font-weight: 300;"> UNIQUE</div>
                                   
                                </div>
                            </div>
                       
            <div style="padding: 0px 10px 0px; width: 300px;">
                        <div class="table_tab_pic" style="width: 300px; height: 50px; font-size: 10px; margin-bottom: 0px;">
                            <div style="text-align: center; padding: 0px 5px; color: #a9a9a9; font-weight: 300; height: 40px;">
                                <img class="img-thumb-mini" src="data:image/png;base64, {pic_mini}" alt="Histogram" style="padding-top: 5px;">
                            </div>
                                <div style="display: flex;">
                                    <div style="width: 50px;"></div>
                                    <div style="width: 50px; font-size: 10px; text-align: left; padding: 0px 5px; color: #a9a9a9; font-weight: 300;"> Unique</div>
                                    <div style="width: 100px; font-size: 11px; text-align: left; padding: 0px 5px; color: {font_colorD}; font-weight: 500;"> {data['DISTINCT']}</div>
                                    <div style="width: 50px;"></div>
                                    <div style="width: 50px; font-size: 10px; text-align: left; padding: 0px 5px; color: #a9a9a9; font-weight: 300;"> Total</div>
                                    <div style="width: 100px; font-size: 11px; text-align: left; padding: 0px 5px; color: #a9a9a9; font-weight: 500;"> {data['TOTAL']}</div>
                                </div>
        
                        </div>
            </div>
        </div>
            """

def wiz(hub,cfg):
    """
    Render a basic plot
    Parameters
    ----------
    hub
        The DataHub containing results from the compute function.
    cfg
        Config instance
    """
    if hub.variable_type == "categorical":
        variable_data = wiz_cat(hub,cfg)
    elif hub.variable_type == "continuous":
        variable_data = wiz_num(hub,cfg)
    elif hub.variable_type == "text":
        variable_data = wiz_text(hub,cfg)
    elif hub.variable_type == "biv_con":
        variable_data = wiz_biv_con(hub)
    elif hub.variable_type == "biv_con_cat":
        variable_data = wiz_biv_con_cat(hub)
    elif hub.variable_type == "biv_cat":
        variable_data = wiz_biv_cat(hub)
    elif hub.variable_type == "datetime":
        variable_data = wiz_dt(hub, cfg)
    elif hub.variable_type == "wrapper":
        variable_data = wiz_dash(hub, cfg)
    elif hub.variable_type == "correlation":
        variable_data = wiz_corr(hub, cfg)
        
    # elif hub.variable_type == "geography_column":
    #     variable_data = render_geo(hub, cfg)
    return variable_data
