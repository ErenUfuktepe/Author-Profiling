import os
import nltk
import pandas as pd
from os.path import exists
import xml.etree.cElementTree as et
import re
import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim_models

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('words')

stop_words = stopwords.words('english')
words = set(nltk.corpus.words.words())


TRUTH_FILE = os.path.join('en', 'truth.txt')
TRUTH_CSV = 'truth.csv'
COLUMNS = ['file', 'gender', 'country']
FEMALE_WORDS = []
MALE_WORDS = []

if exists(TRUTH_CSV):
    truth_dataframe = pd.read_csv(TRUTH_CSV)
else:
    truth_dataframe = pd.read_csv(TRUTH_FILE, delimiter=':::', header=None, names=COLUMNS)
    truth_dataframe.to_csv(TRUTH_CSV, index=False)


def extract_text_from_file(file_name):
    file_path = os.path.join('en', file_name + '.xml')
    documents = et.parse(file_path).find('documents')
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
    return text


def clean_text(text):
    # Removing punctuations
    data_words = gensim.utils.simple_preprocess(str(text), deacc=True)
    # Removing stop words
    data_words = [[word for word in simple_preprocess(str(doc))
                    if word not in stop_words] for doc in data_words]
    return [word for word in data_words if word]


def extract_male_and_female_words(row):
    global MALE_WORDS, FEMALE_WORDS
    text = extract_text_from_file(row['file'])
    word_list = clean_text(text)
    if row['gender'] == 'male':
        MALE_WORDS = MALE_WORDS + word_list
    else:
        FEMALE_WORDS = FEMALE_WORDS + word_list


def create_report(word_list, gender):
    # Create Dictionary
    id2word = corpora.Dictionary(word_list)
    # Create Corpus
    texts = word_list
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=10)
    LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(LDAvis_prepared, 'ldavis_report_' + str(gender) + '.html')


# https://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing
if __name__ == '__main__':
    truth_dataframe.apply(extract_male_and_female_words, axis=1)

    create_report(MALE_WORDS, 'male')
    create_report(FEMALE_WORDS, 'female')
