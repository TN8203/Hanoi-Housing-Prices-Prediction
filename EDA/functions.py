import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def basic_information(df, keys = [], print_dtype = False):
    
    '''
    Function to print some basic information of dataframe: shape, numbers of duplicate values
                                                           number of unique value in each ID column
    Inputs:
    - df: input DataFrame 
    
    - key: a list include feature that need to be checked
    
    - print_dtype: bool, default = False. Whether to print dtype of each column or not
    
    
    '''
    
    print(f'The shape: {df.shape}')
    print('-'*100)
    print(f'Number of duplicate values: {df.shape[0] - df.duplicated().shape[0]}')
    print('-'*100)
    if len(keys) > 0:
        for key in keys:
            print(f'Number of unique {key} are: {len(df[key].unique())}')
    else:
        pass


def count_missing_values(df):
    
    '''
    Function to count missing values in a DataFrame

    Inputs:
    - df (pd.DataFrame): Input DataFrame.

    Outputs:
    - df_nan: A dataframe has two columns: the numbers of missing value and the percentage of missing values which indexs
             are the column name of original dataframe
    '''
    
    total_nan = df.isnull().sum()
    percent_nan = 100 * df.isnull().sum() / len(df) 
    
    df_nan = pd.concat([total_nan, percent_nan], axis=1)  
    df_nan = df_nan.rename(columns = {0: 'total_nan', 1: 'percent_nan'})
    
    df_nan = df_nan[df_nan.iloc[:,1] != 0].sort_values(by = 'percent_nan', ascending=False)
    print(f"Data frame has {df.shape[1]} columns.\nThere are {df_nan.shape[0]} columns that have missing values." )
    return df_nan

def plot_missing_values_percent(df, tight_layout = True, figsize = (20,8), grid = False):
    
    '''
    Function to plot Bar Plots of missing value percentages for each column in a dataframe
    
    Inputs:   
    - df: a DataFrame 
    
    - tight_layout: bool, default = True. Whether to keep tight layout or not
    
    - figsize: tuple, default = (20,8) Figure size of plot 
    
    - grid: bool, default = False. Whether to draw gridlines to plot or not
    
    '''
    df_nan = count_missing_values(df).reset_index()
    if df_nan['percent_nan'].sum() != 0:
        fig, ax = plt.subplots(figsize=figsize, tight_layout=tight_layout)
        sns.barplot(x= 'percent_nan', y = 'index', data = df_nan[['index','percent_nan']], palette='Blues_r')
        ax.set_xlabel('')
        ax.set_ylabel('')  
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_xticks([])
        for bar in ax.containers:
            ax.bar_label(bar, fmt='%.2f%%', fontsize=14)
        ax.spines[['left','right','bottom','top']].set_visible(False)
        ax.tick_params(left = False)
#         ax.set_title('Percentage of NaN values in each column', size=20)
        plt.show()
    else:
        print(f"The dataframe does not contain any NaN values.")
    
def plot_categorical_variables(df, column_name, figsize = (10, 6)):
    
    '''
    Function to plot Categorical Variables Bar Chart and Pie Chart
    
    Inputs:
    - df: DataFrame. The DataFrame from which to plot
    
    - column_name: str. Column's name whose distribution is to be plotted
    
    - figsize: tuple, default = (10,6). Size of the figure to be plotted
    
    '''
    print(f"Total Number of unique categories of {column_name} = {len(df[column_name].unique())}")
    plt.figure(figsize = figsize, tight_layout = False)
    sns.set(style = 'whitegrid', font_scale = 1.2)
    
    #plotting overall distribution of category
    data_to_plot = df[column_name].value_counts()
    df_to_plot = pd.DataFrame({column_name: data_to_plot.index, 'Number of observations': data_to_plot.values})
    
    filtered_values = [value for value in data_to_plot.values if (value / sum(data_to_plot.values)) > 0.07]
    filtered_labels = [label for label, value in zip(data_to_plot.index, data_to_plot.values) if (value / sum(data_to_plot.values)) > 0.07]
    if any(value / sum(data_to_plot.values) < 0.07 for value in data_to_plot.values):    
        filtered_values.append(sum(value for value in data_to_plot.values if (value / sum(data_to_plot.values)) < 0.07))
        filtered_labels.append('Others')
    
    plt.subplot(1, 2, 1)
    s1 = sns.barplot(x = 'Number of observations', y = column_name, data = df_to_plot)
    #s1.set_yticklabels(s1.get_yticklabels(),rotation = 90)
    plt.title(f'Distribution of {column_name}', pad = 20)
    plt.subplots_adjust(wspace=0.3) 

    plt.subplot(1, 2, 2)
    colors = sns.color_palette('pastel')
    s2 = plt.pie(filtered_values, labels=filtered_labels, colors = colors, autopct='%.1f%%', startangle=140)
    
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()

def plot_numerical_variables(df, column_name, figsize = (20,8), hist_plot = True, box_plot = True):
    
    '''
    Function to plot continuous variables distribution
    
    Inputs:
    - df: DataFrame. The DataFrame from which to plot.
    
    - column_name: str. Name of column whose distribution is to be plotted.
    
    - figsize: tuple, default = (20,8). Size of the figure to be plotted.
    
    - hist_plot: bool, default = True. Whether to plot histogram chart for column
    
    - box_plot: bool, default = True. Whether to plot box plot to analyze the whole range of values in continuous variable or not
    
    - number_of_subplots: int. Total number of chart want to be plotted.
    
    '''
    plt.figure(figsize = figsize)
    sns.set_style('whitegrid')
    sns.color_palette("RdBu", 10)

    if hist_plot:
        plt.subplot(1, 2, 1)
        plt.subplots_adjust(wspace=0.5)   
        plt.title("Distribution of %s" %column_name)
        sns.distplot(df[column_name].dropna(),color='red', kde=True,bins=100)
        
    if box_plot:  
        plt.subplot(1, 2, 2)
        plt.subplots_adjust(wspace=0.5) 
        
        sns.boxplot(y=column_name, data=df)
        plt.title("Box-Plot of {}".format(column_name))

    plt.show()    

