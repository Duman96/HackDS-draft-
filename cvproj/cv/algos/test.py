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
import pickle


train_file = pd.read_csv("main.csv", delimiter=",")

train_file['label'] = 'train'

train_file.head()

tags = train_file["result"].values.tolist()

features = train_file["resume"].values.tolist()

processed_features = []

default_path = "/home/seemsred/Desktop/Hackathon DS/HackDS/cvproj/cv/"

upload_path = default_path + "algos/uploads/test/"

stop_words = set(stopwords.words('english'))

print(len(features))


# print(confusion_matrix(y_test,predictions))
# print("1")
# print(classification_report(y_test,predictions))
# print("2")
# print(accuracy_score(y_test, predictions))
# print("3")
# print(mean_absolute_error(predictions, y_test))


def read(file):
    text = textract.process(file)
    return text.decode('utf-8')


def extract(cv):
    text = read(cv)
    text = str(text)
    text = text.replace("\n", " ")
    text = text.lower()
    return text


def parse():
    files = [join(upload_path, f) for f in listdir(upload_path) if isfile(join(upload_path, f))]
    i = 0
    temp_files = []
    # database = pd.DataFrame()
    skills = "mba office logistics english business analysis analytics purchase"
    while i < len(files):
        file = files[i]
        temp_files.append(files[i])
        dat = extract(file)
        lang = detect(dat)
        if lang == "ru":
            trans = TextBlob(dat)
            dat = trans.translate(from_lang='ru', to='en')
        i += 1
        data = str(dat)
        dataw = data.translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.word_tokenize(dataw)
        words = [word for word in tokens if word.isalpha()]
        words = [w for w in words if not w in stop_words]
        words = TreebankWordDetokenizer().detokenize(words)
        # print(dat)
        write_file(str(i), skills, 0, words)
    return temp_files


def write_file(id, skills, result, resume):
    csv = open("uploads/main.csv", "a")
    row = "\n" + id + "," + skills + "," + str(result) + "," + resume
    csv.write(row)


# columns = 'id' + "," + 'skills' + "," + 'result' + "," + 'resume'
# f.write(columns)

temp_files = parse()

#cv = cross_val_score(text_classifier, processed_features, tags, cv=5)

#print(cv.mean())
test_file = pd.read_csv("uploads/main.csv", delimiter=",")


#1
for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)


tagstest = test_file["result"].values.tolist()

featurestest = test_file["resume"].values.tolist()

test_features = []


#2
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


# vectorizer = TfidfVectorizer(max_features=100, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
# processed_features = vectorizer.fit_transform(processed_features).toarray()
#
#
# X_train, X_test, y_train, y_test = train_test_split(processed_features, tags, test_size=0.3, random_state=0)
#
#
# text_classifier = RandomForestClassifier(n_estimators=200)
#
#
# vectorizer1 = TfidfVectorizer(max_features=100, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
# test_features = vectorizer1.fit_transform(test_features).toarray()
# print("123ew")
# print(len(test_features[1]))
#
# #print(temp_files[1])
#
#
# text_classifier.fit(X_train, y_train)
#
# filename = 'finalized_model.sav'
# pickle.dump(text_classifier, open(filename, 'wb'))
#
# # some time later...
#
# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
#
#
# print(len(X_train[1]))
#
# # text_classifier.predict(X_test)
#
# # print(text_classifier.predict(test_features))
# print(len(processed_features))
# print(len(test_features))

#result = loaded_model.score(test_features, tagstest)

#print(result)