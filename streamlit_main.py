"""
# My first app
Here's our first attempt at using data to create a table:
"""

from ssl import PROTOCOL_TLSv1_2
import streamlit as st
import pandas as pd
import altair as alt
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus.reader import pl196x
import re
from wordcloud import WordCloud
import seaborn as sns
import numpy as np
from nltk.util import ngrams
from collections import Counter
from statistics import mean
from PIL import Image



def clean_me(tweet):
    sentence=str(tweet) ##convert all into strings
    sentence = sentence.lower() ##make all of the text lowercase
    sentence = re.sub(r'http\S+', '', sentence)

    tokenizer = RegexpTokenizer(r'\w+') # regex, \w+ is any word character capital, lowercase or numbers
    tokens = tokenizer.tokenize(sentence) #Tokenizers divide strings into lists of substrings
    
    ##this removes any stopwords
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')] 
    
    joined_filtered_words = " ".join(filtered_words)
    
    return joined_filtered_words #tokens, stem_words, lemma_words, joined_filtered_words, filtered_words

def documentNgrams(documents, size):
    ngrams_all = []
    for document in documents:
        tokens = document.split()
        if len(tokens) <= size:
            continue
        else:
            output = list(ngrams(tokens, size))
        for ngram in output:
            ngrams_all.append(" ".join(ngram))
    cnt_ngram = Counter()
    for word in ngrams_all:
        cnt_ngram[word] += 1
    df = pd.DataFrame.from_dict(cnt_ngram, orient='index').reset_index()
    df = df.rename(columns={'index':'words', 0:'count'})
    df = df.sort_values(by='count', ascending=False)
    df = df.head(15)
    df = df.sort_values(by='count')
    return(df)


data = pd.read_csv('content.csv')
data_sentiment = data[['date','sentiment_number','sentiment_label']]
data_sentiment['date'] = pd.to_datetime(data_sentiment['date'])

dates = data_sentiment['date']
data['cleaned'] = data['content'].apply(clean_me)


def average_me(column):
    pos_likes = []
    neg_likes = []
    neu_likes = []

    for x in range(len(data['sentiment_label'])): 
        if data['sentiment_label'][x] =='Positive': 
            pos_likes.append(data[column][x])
        elif data['sentiment_label'][x] =='Negative': 
            neg_likes.append(data[column][x])
        else: 
            neu_likes.append(data[column][x])

    pos_average = mean(pos_likes)
    neg_average = mean(neg_likes)
    neu_average = mean(neu_likes)
    return pos_average,neg_average,neu_average



def styling(column_name, styled_column, col):
    pos_like_ave, neg_like_ave, neu_like_ave = average_me(column_name)

    if (pos_like_ave > neg_like_ave) and (pos_like_ave > neu_like_ave):
        col.header(styled_column)
        styled_column = styled_column.lower()
        col.write(f':sparkles: Average {styled_column} for a positive tweet: {pos_like_ave} :sparkles:')
        col.write(f'Average {styled_column} for a negative tweet: {neg_like_ave}')
        col.write(f'Average {styled_column} for a neutral tweet: {neu_like_ave}')
    elif (neg_like_ave > neu_like_ave) and (neg_like_ave > pos_like_ave):
        col.header(styled_column)
        styled_column = styled_column.lower()
        col.write(f'Average {styled_column} for a positive tweet: {pos_like_ave} ')
        col.write(f':sparkles: Average {styled_column} for a negative tweet: {neg_like_ave} :sparkles:')
        col.write(f'Average {styled_column} for a neutral tweet: {neu_like_ave}')
    else: 
        col.header(styled_column)
        styled_column = styled_column.lower()
        col.write(f'Average {styled_column} for a positive tweet: {pos_like_ave} ')
        col.write(f'Average {styled_column} for a negative tweet: {neg_like_ave} ')
        col.write(f':sparkles: Average {styled_column} for a neutral tweet: {neu_like_ave} :sparkles:')
        



st.set_page_config(layout="wide")

st.title('Twitter Analysis')
def basic_skeleton() -> tuple:
    """Prepare the basic UI for the app"""
    st.sidebar.title('Progress')
    # option = st.selectbox(
    #     'How would you like to be contacted?',
    #     ('PayPal Holdings, Inc.','Moderna, Inc.'))
    option = st.sidebar.text_input('Please enter twitter username', ''
    )

    return option

def main():
    """Central wrapper to control the UI"""
    # add high level site inputs
    text_input = basic_skeleton()
    if text_input:

        # my_bar_graph_one = st.sidebar.progress(0)

        # for percent_complete in range(100):
        #     time.sleep(0.1)
        #     my_bar_graph_one.progress(percent_complete + 1)
    
        
        st.write('The twitter username you have chosen is: ',text_input)
        st.write('The total number of tweets', len(data))
        # st.line_chart(data_sentiment.rename(columns={'date':'index'}).set_index('index'))

        col1,col15,col155, col2,col25,col255 = st.columns((0.5,2,0.5,0.5,2,0.5))
        styling('likeCount', 'Likes',col15)
        styling('retweetCount', 'Retweets',col25)

        col3, col4 = st.columns((2,2))
        col15.header('Sentiment over time')

        c = alt.Chart(data_sentiment).mark_circle().encode(
        x=alt.X('date', axis=alt.Axis( title='Date')),
        y=alt.Y('sentiment_number', axis=alt.Axis(title='Sentiment')),color='sentiment_label')

        # x='date', y='sentiment_number')
        col3.altair_chart(c, use_container_width=True)

                # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        pos = 0
        neg = 0
        neu =0
        for x in data_sentiment['sentiment_label']:
            if x == 'Positive':
                pos +=1
            elif x == 'Negative':
                neg +=1
            else:
                neu +=1
    
        labels = ['Positive', 'Negative', 'Neutral']
        sizes =[pos,neg,neu]
        # fig1, ax1 = plt.subplots()
        # ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        #         shadow=False, startangle=90)
        # ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        

        # st.plotly_chart(fig1)
        fig = go.Figure(
            go.Pie(
            labels = labels,
            values = sizes,
            hoverinfo = "label+percent",
            textinfo = "value"
        ))

        

        col25.header("Pie chart")
        col4.plotly_chart(fig)
        # st.write(list_of_things)
        # st.write(list_of_things[0][0])
        # # unique_string=(" ").join(list_of_sep)
        st.header('WordCloud')
        col5, col6,col7 = st.columns((1,2,1))

        unigrams=documentNgrams(data['cleaned'],1)
        bigrams=documentNgrams(data['cleaned'],2)
        trigrams=documentNgrams(data['cleaned'],3)
        

        wordcloud = WordCloud(mode = "RGBA", background_color=None,width = 1000, height = 500).generate(str(unigrams['words']))
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        col6.pyplot()



        st.header(f'What has @{text_input} been talking about')
        figsns = plt.figure(figsize=(15,5))
        plt.subplots_adjust(wspace=1)
        ax = figsns.add_subplot(131)
        ax.barh(np.arange(len(unigrams['words'])), unigrams['count'], align='center', alpha=.5)
        ax.set_title('Unigrams')
        plt.yticks(np.arange(len(unigrams['words'])), unigrams['words'])
        plt.xlabel('Count')

        ax2 = figsns.add_subplot(132)
        ax2.barh(np.arange(len(bigrams['words'])), bigrams['count'], align='center', alpha=.5)
        ax2.set_title('Bigrams')
        plt.yticks(np.arange(len(bigrams['words'])), bigrams['words'])
        plt.xlabel('Count')

        ax3 = figsns.add_subplot(133)
        ax3.barh(np.arange(len(trigrams['words'])), trigrams['count'], align='center', alpha=.5)
        ax3.set_title('Trigrams')
        plt.yticks(np.arange(len(trigrams['words'])), trigrams['words'])
        plt.xlabel('Count')

        st.pyplot(figsns)








        
 





        

main()

