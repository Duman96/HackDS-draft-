import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from os import listdir
from os.path import isfile, join
from textblob import TextBlob
import string
from nltk.tokenize.treebank import TreebankWordDetokenizer
from langdetect import detect
import textract


filename = 'finalized_model.sav'

loaded_model = pickle.load(open(filename, 'rb'))

test_file = pd.read_csv("uploads/main.csv", delimiter=",")

tagstest = test_file["result"].values.tolist()

featurestest = test_file["resume"].values.tolist()

test_features = []


for sentence in range(0, len(featurestest)):
    # Remove all the special characters
    test_feature = re.sub(r'\W', ' ', str(featurestest[sentence]))

    # remove all single characters
    test_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', test_feature)

    # Remove single characters from the start
    test_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', test_feature)

    # Substituting multiple spaces with single space
    test_feature = re.sub(r'\s+', ' ', test_feature, flags=re.I)

    # Removing prefixed 'b'
    test_feature = re.sub(r'^b\s+', '', test_feature)

    # Converting to Lowercase
    test_feature = test_feature.lower()

    test_features.append(test_feature)

vectorizer1 = TfidfVectorizer(max_features=100, stop_words=stopwords.words('english'))
test_features = vectorizer1.fit_transform(test_features).toarray()
# print(test_features)
print(loaded_model.predict(test_features))
