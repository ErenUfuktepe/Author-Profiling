import csv
import os
import xml.etree.cElementTree as ET
import re
import string
import nltk
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from nltk import word_tokenize


# Remove the below comments in the first run.

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


# Method takes a directory as an input and returns the number of files in given directory
def number_of_files(path):
    files_in_directory = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
    return files_in_directory


# Method takes 2 two parameters as an input which are file name and path
# an checks weather the given file is available in given directory
# file_name = file name that you want to look for
# path = directory that want to search for
def check_file(file_name, path):
    for index in range(number_of_files(path)):
        if os.listdir(path)[index] != file_name:
            return True
        else:
            return False


# Method takes one parameter as an input which is a directory to look for
# and return weather the truth file which shows the files gender.
# It first check weather the file is available in the given directory. if
# it is available in the given directory it returns the file otherwise it
# will return false.
def truth_file_path(path):
    if check_file('truth.txt', path):
        file = (os.path.join(path, 'truth.txt'))
        return file
    else:
        return False


# Method takes one parameter as an input which is directory to look for
# and returns the list of tweet files inside the directory.
def create_file_list(path):
    my_files = []
    for index in range(number_of_files(path)):
        if os.listdir(path)[index] != 'truth.txt':
            my_files.append(os.listdir(path)[index])
    return my_files


# Method takes two parameter as an input which are path and my_files and returns
# the files with their directories. Path is the directory of the files and the my_files
# is the list of files. For example, directory is 'C:\Users\name\Desktop\en' and one of
# the file name in the list is '1ac7a7acc78338e721e8f43268402587.xml' then it will combine both
# like 'C:\Users\name\Desktop\en\1ac7a7acc78338e721e8f43268402587.xml'. It will do this operation
# for all files then return them as list.
def create_file_list_with_paths(path, my_files):
    my_list = []
    for file in my_files:
        my_list.append(os.path.join(path, file))
    return my_list


# Method takes one parameter as an input which is truth file and returns a list which
# includes females and males files.
def gender_partition(truth_file):
    text_file = open(truth_file, "r")
    lines = text_file.readlines()
    male_files = []
    female_files = []
    for index in lines:
        tmp = index.split(':::')
        if tmp[1] == 'female':
            female_files.append(tmp[0] + ".xml")
        if tmp[1] == 'male':
            male_files.append(tmp[0] + ".xml")
    return [male_files, female_files]


def tweet_analysis(files, male_list, female_list):
    csv_file = open('TweetData.csv', 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['gender', 'prepositions', 'determiners', 'pronouns', 'text length', 'symbol', 'conjunction',
                         'cardinal number', 'past tense'])

    tweets_file = open('tweets.txt', "w+", encoding="utf8")

    number_of_total_files = len(male_list) + len(female_list)
    file_paths = create_file_list_with_paths(xml_file_path, files)

    for i in range(number_of_total_files):
        root = ET.parse(file_paths[i]).find('documents')
        temp = files[i]
        gender = 1
        for c in root.iterfind('document'):
            temp_list = []
            remove = string.punctuation.replace("-", "")
            pattern = r"[{}]".format(remove)
            my_text = re.sub(pattern, "", c.text)
            text = word_tokenize(my_text)
            tweets_file.write(my_text)

            tags = nltk.pos_tag(text)
            prepositions_counter = 0
            determiners_counter = 0
            pronouns_counter = 0
            symbol_counter = 0
            conjunction_counter = 0
            cardinal_number_counter = 0
            past_tense_counter = 0

            for tag in tags:
                # Preposition
                if tag[1] == 'IN' or tag[1] == "TO":
                    prepositions_counter = prepositions_counter + 1
                # Determiner
                if tag[1] == 'DT' or tag[1] == 'PDT' or tag[1] == 'WDT':
                    determiners_counter = determiners_counter + 1
                # Pronoun
                if tag[1] == 'NNS' or tag[1] == 'PRP' or tag[1] == 'PRP$' or tag[1] == 'WP' or tag[1] == 'NN' \
                        or tag[1] == 'WP$':
                    pronouns_counter = pronouns_counter + 1
                # Symbol
                if tag[1] == 'SYM':
                    symbol_counter = symbol_counter + 1
                # Conjunction
                if tag[1] == 'CC':
                    conjunction_counter = conjunction_counter + 1
                # Cardinal Number
                if tag[1] == 'CD':
                    cardinal_number_counter = cardinal_number_counter + 1
                # Past Tense
                if tag[1] == 'VBD':
                    past_tense_counter = past_tense_counter + 1

            for file_name in females:
                if temp == file_name:
                    gender = 0
                    break

            temp_list.append(gender)
            temp_list.append(prepositions_counter)
            temp_list.append(determiners_counter)
            temp_list.append(pronouns_counter)
            temp_list.append(len(my_text))
            temp_list.append(symbol_counter)
            temp_list.append(conjunction_counter)
            temp_list.append(cardinal_number_counter)
            temp_list.append(past_tense_counter)
            csv_writer.writerow(temp_list)

    csv_file.close()
    tweets_file.close()


xml_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "en\\")

print("Listing files...")
file_list = create_file_list(xml_file_path)

if truth_file_path(xml_file_path):
    print("Gender partitioning...")
    males = gender_partition(truth_file_path(xml_file_path))[0]
    females = gender_partition(truth_file_path(xml_file_path))[1]
else:
    print("Could not find truth file.")
    exit()


print("Analysing tweets...")
tweet_analysis(file_list, males, females)

csv_read = pd.read_csv('TweetData.csv')
inputs = np.array(csv_read[["prepositions", "determiners", "pronouns", "text length", "symbol", "conjunction",
                            "cardinal number", "past tense"]])
output = np.array(csv_read['gender'])
# Prepare dataset
train_x, test_x, train_y, test_y = train_test_split(inputs, output, train_size=0.3, test_size=0.7)
# Create and configure model
model = LogisticRegression(solver='lbfgs', multi_class='auto')
# Fit model
model.fit(train_x, train_y)
score = cross_val_score(model, test_x, test_y, cv=10)
predicted_score = sum(score) / float(len(score))

print("Accuracy is: ", predicted_score * 100)
