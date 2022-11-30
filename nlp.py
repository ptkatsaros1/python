import pandas as pd
import gzip
import re
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.formula.api import logit
from statsmodels.formula.api import mnlogit
from nltk.corpus import stopwords
import math
from wordcloud import STOPWORDS
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import calendar
import statsmodels.api as sm

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def add_columns(df):
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['year'] = pd.DatetimeIndex(df['date']).year
    return df

def get_word_count(df):
    df['comment'] = df.comment.replace('[^a-zA-Z ]', '', regex=True)
    df['comment'] = df.comment.replace(r'\s+', ' ', regex=True)
    df['comment'] = df.comment.str.strip()
    df['word_count'] = df['comment'].str.count(r'\s')

    for i in range (len(df.index)):
        if df['word_count'][i] > 0:
            df['word_count'][i] = df['word_count'][i]+1
    df['word_count'] = df['word_count'].fillna(0)

    return df

def sentiment_pol(comment):
    return TextBlob(comment).polarity

def sentiment_type_calc(df):
    df['sentiment_type'] = 0
    for i in range (len(df.index)):
        if df['sentiment_score'][i] > 0:
            df['sentiment_type'][i] = 'positive'
        elif df['sentiment_score'][i] == 0:
            df['sentiment_type'][i] = 'neutral'
        else:
            df['sentiment_type'][i] = 'negative'
    return df

def descriptive_stats(df):
    print(df[['rating', 'sentiment_score', 'word_count']].describe())

def pearsonr(df):
    correlationb, p_valueb = stats.pearsonr(df['sentiment_score'], df['word_count']) 
    print(f'\n\nCorrelation: {correlationb.round(4)}\nP-Value: {p_valueb.round(4)}')
    print('\nThe direction is negative, the p-value is signifigant and the correlation is somewhat weak.')

def word_count_squared(df):
    df['word_count_square'] = df['word_count']**2
    return df

def product_check(df):
    product_list = []
    my_file = open('products.txt', 'r')

    product_list = list(my_file)
    product_list = [s.replace('\n', '') for s in product_list]
    pattern = '|'.join(product_list)

    my_file.close()

    df['comment'] = df['comment'].str.lower()
    df['product_present'] = df['comment'].str.contains(pattern)

    return df

def ols_reg(df):
    ols_model = ols('sentiment_score~rating+product_present+word_count+word_count_square', df).fit()
    df['predict_sentiment'] = ols_model.predict(df)
    print(ols_model.summary())

def graphs(df):
    
    plt.figure()
    plt.scatter(df['word_count'], df['sentiment_score'], 
                s=1, color='blue')
    plt.scatter(df['word_count'], df['predict_sentiment'], 
                s=1, color='red')
    plt.xlabel('Word Count')
    plt.xlim([0, 1000])
    plt.ylabel('Sentiment Score')
    plt.title('Word Count v. Sentiment Score')
    plt.show()

    df2 = df.groupby('month', as_index=False)['rating'].value_counts()
    df2 = df2.groupby('month', as_index=False)['count'].sum()
    d = dict(enumerate(calendar.month_name))
    df2['month'] = df2['month'].map(d)
    plt.bar(df2['month'], df2['count'], width=.8)
    df2['count'].plot(color='red')
    plt.title('Number of Reviews by Month')
    plt.yticks(np.arange(0,5500,500))
    plt.xlabel('Month')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()

    df3 = df.groupby('sentiment_type', as_index = False)['rating'].value_counts()
    df_pivot = pd.pivot_table(
    df3, 
    values="count",
    index="rating",
    columns="sentiment_type", 
    aggfunc=np.mean
    )
    ax = df_pivot.plot(kind="bar")
    plt.title('Sentiment Type Distribution Across Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Number of Reviews')
    plt.show()

    df4 = df.groupby('rating', as_index = False)['sentiment_score'].mean()

    plt.bar(df4['rating'], df4['sentiment_score'], width=.8)
    plt.title('Average Sentiment Score Across Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Average Sentiment Score')
    plt.show()


def word_freq_analysis(df):
    df = df[df['year'] >= 2010]
    stopword_list1 = list(STOPWORDS)
    stopword_list2 = list(stopwords.words('english'))
    stopword_list = stopword_list1 + stopword_list2
    stopword_list = list(set(stopword_list))
    
    word_corpus = []
    
    for comment in df['comment']:
        comment = comment.lower()
        comment = re.sub("[^a-z'\s]", '', comment)
        comment = re.sub('\s+', ' ', comment)
        comment = comment.strip()
        word_list = comment.split(' ')
        
        for word in word_list:
            if word not in stopword_list:
                    stem_word = PorterStemmer().stem(word)
                    word_corpus.append(stem_word)
                    
    word_series = pd.Series(word_corpus)
    word_df = word_series.value_counts().to_frame().reset_index()

    word_df.columns = ['stem_word', 'count']
    
    top_words = word_df[:25]

    plt.figure()
    plt.title("Top Words Used")
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.bar(top_words['stem_word'], top_words['count'])
    plt.xticks(rotation=75)
    plt.show()

#extra 1
def bar_month(df):
    df_month = df
    d = dict(enumerate(calendar.month_name))
    df_month['month'] = df_month['month'].map(d)
    df_month = df_month.groupby('month', as_index=False)['rating'].mean()

    plt.bar(df_month['month'], df_month['rating'], width=.8)
    plt.title('Average Reviews by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Review')
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()
    print('\n\nThere is no discernable change in average rating by month.\n\n')

def regression(df):
    x = df[['word_count', 'sentiment_score']]
    y = df['rating']

    x = sm.add_constant(x)
    model = sm.OLS(y,x).fit()
    predictions = model.predict(x)

    print_model = model.summary()
    print(print_model)
def pie(df):
    list1=[]
    list1 = (df['sentiment_type'].value_counts(normalize=True) * 100)
    labels1 = 'positive', 'negative', ''
    
    explode = (0, 0.1, 0)  
    fig1, ax1 = plt.subplots()
    ax1.pie(list1, explode=explode, labels=labels1, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()


def main(): 
    df = getDF('reviews_Office_Products_5.json.gz')

    df = df.rename(columns={'reviewText':'comment', 'overall':'rating', 'reviewTime':'date'})

    df = add_columns(df)
    df = get_word_count(df)
    df['sentiment_score'] = df.apply(lambda x: sentiment_pol(x.comment), axis=1)
    df = sentiment_type_calc(df)
    descriptive_stats(df)
    pearsonr(df)
    df = word_count_squared(df)
    df = product_check(df)
    ols_reg(df)
    graphs(df)
    word_freq_analysis(df)

    # ----extras-----
    #1. bar chart of average rating by month
    bar_month(df)
    #2. multiple linear regression
    regression(df)
    #3. pie chart
    pie(df)


main()