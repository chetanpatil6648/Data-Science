#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Assignments
# Amazon Product Reviews of iPhone 11(64gb)


# In[2]:


# Import Libraries
import numpy as np 
import pandas as pd 
import string 
import spacy 

from matplotlib.pyplot import imread
from matplotlib import pyplot as plt
from wordcloud import WordCloud
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Load data sets
reviews = pd.read_csv("\\Users\\Rohit\\Downloads\\ReviewDetails (2).csv")
reviews


# # Text Preprocessing

# In[4]:


import re #regular expression
import string
# Remove Punctuation

def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub("[0-9" "]+"," ",text)
    text = re.sub('[‘’“”…]', '', text)
    return text

clean = lambda x: clean_text(x)


# In[5]:


reviews['text'] = reviews.text.apply(clean)
reviews.text


# In[6]:


reviews = [text.strip() for text in reviews.text] # remove both the leading and the trailing characters
reviews = [text for text in reviews if text] # removes empty strings, because they are considered in Python as False
reviews[0:10]


# In[8]:


# Joining the list into one string/text
reviews_text = ' '.join(reviews)
len(reviews_text)


# In[14]:


print(reviews_text)


# In[15]:


# Tokenization
from nltk.tokenize import word_tokenize
reviews_tokens = word_tokenize(reviews_text)
print(reviews_tokens)


# In[16]:


len(reviews_tokens) 


# In[17]:


# Stopwords
# Remove Stopwords
import nltk
from nltk.corpus import stopwords
my_stop_words = stopwords.words('english')
my_stop_words.append('the')
no_stop_tokens = [word for word in reviews_tokens if not word in my_stop_words]
print(no_stop_tokens[0:100])


# In[18]:


len(no_stop_tokens)  


# In[19]:


# Noramalize the data
lower_words = [text.lower() for text in no_stop_tokens]
print(lower_words[0:50])


# In[20]:


# NLP english language model of spacy library
nlp = spacy.load("en_core_web_sm")


# In[21]:


# lemmas being one of them, but mostly POS, which will follow later
doc = nlp(' '.join(lower_words))
print(doc[0:40])


# In[22]:


lemmas = [token.lemma_ for token in doc]
print(lemmas[0:40])


# # Feature Extaction

# # 1. Using CountVectorizer

# In[23]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
tweetscv=cv.fit_transform(lemmas)


# In[24]:


print(cv.get_feature_names()[100:200])


# In[25]:


print(tweetscv.toarray()[100:200])


# In[26]:


print(tweetscv.toarray().shape) 


# # 2. CountVectorizer with N-grams (Bigrams & Trigrams)

# In[27]:


cv_ngram_range=CountVectorizer(analyzer='word',ngram_range=(1,3),max_features=100)
bow_matrix_ngram=cv_ngram_range.fit_transform(lemmas)


# In[28]:


print(cv_ngram_range.get_feature_names())
print(bow_matrix_ngram.toarray())


# # 3. TF-IDF Vectorizer

# In[29]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv_ngram_max_features=TfidfVectorizer(norm='l2',analyzer='word',ngram_range=(1,3),max_features=500)
tfidf_matix_ngram=tfidfv_ngram_max_features.fit_transform(lemmas)


# In[30]:


print(tfidfv_ngram_max_features.get_feature_names())
print(tfidf_matix_ngram.toarray())


# In[31]:


clean_reviews=' '.join(lemmas)
clean_reviews


# # Generate Word Cloud

# In[32]:


# Define a function to plot word cloud
def plot_cloud(wordcloud):
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud)
    plt.axis('off')
    
# Generate Word Cloud

from wordcloud import WordCloud, STOPWORDS

STOPWORDS.add('pron')
STOPWORDS.add('rt')
STOPWORDS.add('yeah')
wordcloud=WordCloud(width=3000,height=2000,background_color='black',max_words=50,
                   colormap='Set1',stopwords=STOPWORDS).generate(clean_reviews)
plot_cloud(wordcloud)


# # Named Entity Recognition (NER)

# In[33]:


##Part Of Speech Tagging
nlp = spacy.load("en_core_web_sm")

one_block = clean_reviews
doc_block = nlp(one_block)
spacy.displacy.render(doc_block, style='ent', jupyter=True)


# In[34]:


for token in doc_block[500:600]:
    print(token, token.pos_)


# In[35]:


# Filtering the nouns and verbs only
nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs[500:600])


# In[36]:


# Counting the noun & verb tokens
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

X=cv.fit_transform(nouns_verbs)
sum_words=X.sum(axis=0)

words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq, key=lambda x: x[1], reverse=True)

wd_df=pd.DataFrame(words_freq)
wd_df.columns=['word','count']
wd_df[0:10] # viewing top ten results


# In[37]:


# Visualizing results (Barchart for top 10 nouns + verbs)
wd_df[0:10].plot.bar(x='word',figsize=(12,8),title='Top 10 nouns and verbs');


# # Emotion Mining - Sentiment Analysis

# In[38]:


reviews = pd.read_csv("/Users/mayur/Desktop/Excelr Solution/Assignment-11(Text Mining)/iPhone 11 Review2.csv")


# In[39]:


import re #regular expression
import string
# Remove Punctuation

def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub("[0-9" "]+"," ",text)
    text = re.sub('[‘’“”…]', '', text)
    return text

clean = lambda x: clean_text(x)


# In[40]:


reviews['text'] = reviews.text.apply(clean)


# In[45]:


reviews = [text.strip() for text in reviews.text] 
reviews = [text for text in reviews if text] 
reviews


# In[46]:


from nltk import tokenize
sentences = tokenize.sent_tokenize(' '.join(reviews))
sentences


# In[47]:


sent_df = pd.DataFrame(reviews,columns=['sentence'])
sent_df


# In[48]:


affin = pd.read_csv('/Users/mayur/Documents/Afinn.csv.xls', sep=',', encoding='latin-1')


# In[49]:


affinity_scores=affin.set_index('word')['value'].to_dict()
affinity_scores


# In[50]:


# Custom function: score each word in a sentence in lemmatised form, but calculate the score for the whole original sentence
nlp=spacy.load('en_core_web_sm')
sentiment_lexicon=affinity_scores

def calculate_sentiment(text:str=None):
    sent_score=0
    if text:
        sentence=nlp(text)
        for word in sentence:
            sent_score+=sentiment_lexicon.get(word.lemma_,0)
    return sent_score


# In[51]:


# manual testing
calculate_sentiment(text='great')


# In[52]:


# Calculating sentiment value for each sentence
sent_df['sentiment_value']=sent_df['sentence'].apply(calculate_sentiment)
sent_df['sentiment_value']


# In[53]:


sent_df.sort_values(by='sentiment_value')


# In[54]:


# Sentiment score of the whole review
sent_df['sentiment_value'].describe()


# In[55]:


# negative sentiment score of the whole review
sent_df[sent_df['sentiment_value']<=0]


# In[56]:


# positive sentiment score of the whole review
sent_df[sent_df['sentiment_value']>0]


# In[57]:


# Adding index cloumn
sent_df['index']=range(0,len(sent_df))
sent_df


# In[58]:


# Plotting the sentiment value for whole review
import seaborn as sns
plt.figure(figsize=(15,10))
sns.distplot(sent_df['sentiment_value'])


# In[59]:


# Plotting the line plot for sentiment value of whole review
plt.figure(figsize=(15,10))
sns.lineplot(y='sentiment_value',x='index',data=sent_df)

