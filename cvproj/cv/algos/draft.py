import textract
import PyPDF2
import nltk
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import os, sys
from langdetect import detect
from os import listdir
from os.path import isfile, join
from textblob import TextBlob
import string
from nltk.tokenize.treebank import TreebankWordDetokenizer


default_path = "/home/seemsred/Desktop/Hackathon DS/HackDS/cvproj/cv/"

upload_path = default_path + "algos/uploads/test/"

stop_words = set(stopwords.words('english'))


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
    x = 0
    # database = pd.DataFrame()
    skills = "office windows exchange active directory itil atc unix linux it"
    tokenskills = nltk.word_tokenize(skills)
    z = 0
    while i < len(files):
        file = files[i]
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
        y = 0
        while x < len(tokenskills):
            while y < len(words):
                if tokenskills[x] == words[y]:
                    z += 1
    return z

    # print(dat)


q = parse()
print(q)