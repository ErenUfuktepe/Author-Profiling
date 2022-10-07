import os
import xml.etree.cElementTree as et
import re
import nltk
import pandas as pd
import numpy as np
from os.path import exists
import gensim
from gensim.utils import simple_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix

# Remove the below comments in the first run.
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

stop_words = stopwords.words('english')
words = set(nltk.corpus.words.words())

MALE_WORDS = ['trump', 'one', 'like', 'get', 'good', 'new', 'people', 'time', 'via', 'great', 'day', 'would', 'see',
              'today', 'thanks', 'think', 'know', 'us', 'well', 'back', 'right', 'go', 'going', 'year', 'still',
              'love', 'need', 'last', 'first', 'much']

FEMALE_WORDS = ['time', 'think', 'back', 'well', 'still', 'take', 'world', 'tonight', 'days', 'away', 'maybe',
                'everything', 'working', 'job', 'school', 'lot', 'food', 'without', 'left', 'since', 'top', 'forward',
                'anything', 'part', 'hair', 'sleep', 'run', 'absolutely', 'state', 'stay']

TRUTH_FILE = os.path.join('en', 'truth.txt')
TRUTH_CSV = 'truth.csv'
COLUMNS = ['file', 'gender', 'country']


def read_and_clean_text(file):
    documents = et.parse(file).find('documents')
    list_of_tweets = list(map(lambda x: x.text, documents.iterfind('document')))
    merged_tweets = ' '.join(list_of_tweets)
    # Removing words that start with "https://t.co/"
    text = re.sub(r'\bhttps://t.co/\w+', "", merged_tweets)
    # Removing words that start with "@"
    text = re.sub(r'\b@\w+', "", text)
    # Removing words that start with "#"
    text = re.sub(r'\b#\w+', "", text)
    # Removing words that contain _
    text = re.sub(r'[^ ]*_[^ ]*', "", text)
    text = " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in words or not w.isalpha())
    # Removing punctuations
    text = gensim.utils.simple_preprocess(str(text), deacc=True)
    # Removing stop words
    text = [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in text]
    text = [word[0] for word in text if word]

    return text


def count_tag(row, tag_list):
    file = os.path.join('en', row['file'] + '.xml')
    text = read_and_clean_text(file)
    tags = nltk.pos_tag(text)

    return sum([1 for tag in tags if tag[1] in tag_list])


def feature_count(row, feature_list):
    file = os.path.join('en', row['file'] + '.xml')
    text = read_and_clean_text(file)

    return sum([1 for word in text if word in feature_list])


if exists(TRUTH_CSV):
    truth_dataframe = pd.read_csv(TRUTH_CSV)
else:
    truth_dataframe = pd.read_csv(TRUTH_FILE, delimiter=':::', header=None, names=COLUMNS)
    truth_dataframe.to_csv(TRUTH_CSV, index=False)


if exists("dataset.csv"):
    truth_dataframe = pd.read_csv("dataset.csv")
else:
    truth_dataframe['gender'] = truth_dataframe['gender'].map({'male': 0, 'female': 1})
    truth_dataframe = truth_dataframe.drop(columns=['country'])
    truth_dataframe = truth_dataframe.rename(columns={'gender': 'target'})

    truth_dataframe['preposition'] = truth_dataframe.apply(count_tag, axis=1, tag_list=['IN', 'TO'])
    truth_dataframe['determiner'] = truth_dataframe.apply(count_tag, axis=1, tag_list=['DT', 'PDT', 'WDT'])
    truth_dataframe['pronoun'] = truth_dataframe.apply(count_tag, axis=1, tag_list=['NNS', 'PRP', 'PRP$', 'WP', 'NN', 'WP$'])
    truth_dataframe['male_feature_count'] = truth_dataframe.apply(feature_count, axis=1, feature_list=MALE_WORDS)
    truth_dataframe['female_feature_count'] = truth_dataframe.apply(feature_count, axis=1, feature_list=FEMALE_WORDS)

    truth_dataframe.to_csv('dataset.csv', index=False)

# Gender
target = np.array(truth_dataframe['target'])
features = np.array(truth_dataframe[['preposition', 'determiner', 'pronoun',
                                     'male_feature_count', 'female_feature_count']])

train_x, test_x, train_y, test_y = train_test_split(features, target, train_size=0.3, test_size=0.7)
# Create and configure model
model = LogisticRegression(solver='lbfgs', multi_class='auto')
# Fit model
model.fit(train_x, train_y)
score = cross_val_score(model, test_x, test_y, cv=10)
predicted_score = sum(score) / float(len(score))
print("Accuracy is: ", predicted_score * 100)

y_pred = model.predict(test_x)
CM = confusion_matrix(test_y, y_pred)
print("Confusion Matrix:")
print(CM)
