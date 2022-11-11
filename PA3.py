import pandas as pd
import datetime as dt
import numpy as np
import seaborn as sns
import re
from IPython.display import display
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'
# pylint: disable=E1101



def main():
    df = pd.read_csv('reviews_pa3.txt', sep='\t')
    df = df.replace(r'\s+', ' ', regex=True)
    df2 = pd.read_csv('cars_pa3.txt', sep='#')
    
    
    df_new = counterYear(df)
    
    descriptiveStats(df)
    
    df_pivot_1 = get_pivot_1(df_new)
    plot_ratings_by_year(df_pivot_1)
    
    avgcountyear(df_new)
    
    seasonyear(df_new)
    
    year_car_rate(df2, df_new)
    
    price_rate(df2, df_new)
    
    cool_chart(df2, df_new)


def counterYear(df):
    df_new = df

    df_new['comment'] = df.comment.replace('[^a-zA-Z ]', '', regex=True)
    df_new['comment'] = df.comment.replace(r'\s+', ' ', regex=True)
    df_new['comment'] = df.comment.str.strip()
    df_new['word_count'] =df_new['comment'].str.count(r'\s')+1
    df_new['word_count'] = df_new['word_count'].fillna(0)
    df.loc[df["word_count"] == 1.0, "word_count"] = 0
    
    df_new['yearonly'] = pd.DatetimeIndex(df['date']).year
    
    print('\n\n')
    print(df_new)
    print('\n------------------------------------------------------------------------------------\n')
    
    return df_new

def descriptiveStats(df):
    #simple .describe method for descriptive states
    print("Word Count Descriptive Statistics:\n")
    df10 = df['word_count'].describe()[['count','mean','std','min','max']].apply("{0:.2f}".format).reset_index()
    print (df10.to_string(header=None, index=None))
    print('\n------------------------------------------------------------------------------------\n')
    print("Rating Descriptive Statistics:\n")
    df11 = df['rating'].describe()[['count','mean','std','min','max']].apply("{0:.2f}".format).reset_index()
    print (df11.to_string(header=None, index=None))
    print('\n------------------------------------------------------------------------------------\n')

def get_pivot_1(df):
    #get pivot table and get count values
    df_group = df.groupby('rating')['yearonly'].value_counts()
    df_group = df_group.to_frame()
    df_group = df_group.rename({'yearonly':'count'}, axis=1)
    df_group = df_group.reset_index()
    df_group['rating'] = df_group['rating'].astype('int')
    df_pivot = df_group.pivot(index='rating', columns='yearonly', values='count')
    df_pivot = df_pivot.fillna(0)
    
    print(df_pivot)
    print('\n------------------------------------------------------------------------------------\n')
    return df_pivot

def plot_ratings_by_year(df):
    #graph formating
    df.plot(kind='bar')
    plt.title('Number of Ratings by Year')
    plt.xlabel('Ratings')
    plt.ylabel('Count')
    plt.yticks(np.arange(0,6,1))
    plt.xticks(rotation=0)
    plt.legend(loc='upper left')
    plt.show()

def avgcountyear(df):
    ##groupby and get mean
    df['avgcount'] = df.groupby('yearonly')['word_count'].transform('mean')
    #create new df and drop duplicates
    df2 = df[['yearonly', 'avgcount']].copy()
    df2 = df2.drop_duplicates()
    
    dfavg = df2.set_index('yearonly')
    
    print(dfavg)
    print('\n------------------------------------------------------------------------------------\n')
    
    #plot graph
    df2.plot(x="yearonly", y=["avgcount"], kind="bar")
    plt.title('Average Word Count by Year')
    plt.xlabel('Year')
    plt.ylabel('Word Count')
    plt.yticks(np.arange(0,6,.5))
    plt.xticks(rotation=0)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
        
def seasonyear(df):
    df['months'] = pd.DatetimeIndex(df['date']).month
    
    #create conditions for assignment
    conditions1 = [
    (df['months'] >= 1) & (df['months'] <= 4), 
    (df['months'] >= 5) & (df['months'] <= 8),
    (df['months'] >= 9) & (df['months'] <= 12)
    ]

    # values list to assign
    values1 = [df.yearonly.astype(str)+'1', df.yearonly.astype(str)+'2', df.yearonly.astype(str)+'3']

    # use np.select to assign based on conditions
    df['yearseason'] = np.select(conditions1, values1)

    #create conditions for assignment
    conditions = [
    (df['yearseason'] == '20191'), 
    (df['yearseason'] == '20192'),
    (df['yearseason'] == '20193'),
    (df['yearseason'] == '20201'), 
    (df['yearseason'] == '20202'),
    (df['yearseason'] == '20203'),
    (df['yearseason'] == '20211'), 
    (df['yearseason'] == '20212'),
    (df['yearseason'] == '20213'),
    ]

    # values list to assign
    values = ['Winter 19', 'Summer 19', 'Fall 19','Winter 20', 'Summer 20', 'Fall 20','Winter 21', 'Summer 21', 'Fall 21']

    # first, sort index for chart purposes, then use np.select to assign based on conditions, copy into new df and drop duplicates
    df.index = np.select(conditions, values)
    
    df4 = df
    #drop any empty strings that will avoid nan drop and mess with count
    df4['comment'].replace('', np.nan, inplace=True)
    df4 = df4.dropna()
    #insert count column then simplify table with only 2 columns and index, drop the duplicates
    df4['count'] = df4.index.value_counts()
    df4 = df4[['yearseason', 'count']].copy()
    df4 = df4.drop_duplicates()
    
    #list with all the codes
    check = ['20191', '20192', '20193', '20201', '20202', '20203', '20211', '20212', '20213']
    
    #if one of the codes doesnt exist in the df, it wont have a count of zero, so retroactively add it and assign count 0
    for i in range(0,9):
        if (df4['yearseason'].eq(check[i])).any() == False:
            df4 = df4.append({'yearseason':check[i], 'count':0}, ignore_index=True)
  
    
    #need to re-assign index values
    conditions = [
    (df4['yearseason'] == '20191'), 
    (df4['yearseason'] == '20192'),
    (df4['yearseason'] == '20193'),
    (df4['yearseason'] == '20201'), 
    (df4['yearseason'] == '20202'),
    (df4['yearseason'] == '20203'),
    (df4['yearseason'] == '20211'), 
    (df4['yearseason'] == '20212'),
    (df4['yearseason'] == '20213'),
    ]
    # values list to assign
    values = ['Winter 19', 'Summer 19', 'Fall 19','Winter 20', 'Summer 20', 'Fall 20','Winter 21', 'Summer 21', 'Fall 21']
    # first, sort index for chart purposes, then use np.select to assign based on conditions, copy into new df and drop duplicates
    df4.index = np.select(conditions, values)
    
    #sort for graphing purposes
    df4 = df4.sort_values(by=['yearseason'])
    
    print(df4)
    
    #line graph
    df4.plot()
    plt.title('Number of Reviews by Season and Year')
    plt.xlabel('Year Season')
    plt.ylabel('Count')
    plt.yticks(np.arange(0,6,1))
    plt.xticks(rotation=30)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    
def year_car_rate(df2, df_new):
    
    #new column with cartype that compares reviews file to the car file and maps the matches of name and make
    df_new['cartype'] = df_new['name'].map(df2.set_index('name')['make'])
    #drop null reiews to not interfere with computations
    df_new = df_new.dropna()
    df_new = df_new.drop(df_new[df_new.word_count == 0].index)
    #get averages with .transform and the groupings
    cols =[df_new.cartype, df_new.yearonly]
    df_new['avgrate'] = df_new.groupby(cols)['rating'].transform('mean')
    #put in a new df, drop diplicates and pivot
    df3 = df_new[['yearonly', 'avgrate', 'cartype']].copy()
    df3 = df3.drop_duplicates()
    df3 = df3.pivot(index='yearonly', columns='cartype', values='avgrate')
    print('\n------------------------------------------------------------------------------------\n')
    print(df3)
    
    #graphing    
    ax = df3.plot(kind='bar')
    plt.title('Average Yearly Rating by Car Make')
    plt.xlabel('Year')
    plt.ylabel('Average Rating')
    plt.yticks(np.arange(0,5.5,.5))
    plt.xticks(rotation=0)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    
def price_rate(df2, df_new):
    df_corr = df_new
    df_corr['price'] = df_new['name'].map(df2.set_index('name')['price'])
    df_corr = df_corr.dropna()
    df_corr = df_corr[['rating', 'price']].copy()
    df_corr = df_corr.sort_values(by=['rating'])
    correlation, p_value = stats.pearsonr(df_corr['rating'], df_corr['price'])
    
    print('\n------------------------------------------------------------------------------------\n')
    print("Many believe price affects ratings. Does it?\n\nRunning statistical analysis on the correlation between price and rating......")
    print("\n")
    print(f"Correlation: {correlation}\nP-Value: {p_value}", end='\n')
    print("\nThe P-Value suggests that there is insufficient evidence to support the claim that price affects rating.\nA correlation of .139 also suggests that there is no correlation between the two.\nThe data points can be visually inspected in the scatter plot.", end ='\n\n')
    
    plt.figure()
    plt.scatter(df_corr['rating'], df_corr['price'], marker='.')
    plt.title('Price vs Rating')
    plt.xlabel('Rating')
    plt.ylabel('Price')
    plt.tight_layout()
    plt.show()

def cool_chart(df2, df_new):
    
    df_corr2 = df_new
    df_corr2['price'] = df_new['name'].map(df2.set_index('name')['price'])
    df_corr2['make'] = df_new['name'].map(df2.set_index('name')['manufacture_year'])
    df_corr2 = df_corr2.dropna()
    df_corr2 = df_corr2.drop('avgcount', axis=1)
    
    # Plot
    plt.figure(figsize=(12,10), dpi= 80)
    sns.heatmap(df_corr2.corr(), xticklabels=df_corr2.corr().columns, yticklabels=df_corr2.corr().columns, cmap='RdYlGn', center=0, annot=True)

    # Decorations
    plt.title('Correlogram of Reviews', fontsize=22)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


main()