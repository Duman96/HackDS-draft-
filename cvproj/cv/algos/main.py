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


default_path = "/home/seemsred/Desktop/Hackathon DS/HackDS/cvproj/cv/media/"
path_to_right_sup = default_path + "rightsup"
path_to_right_purch = default_path + "rightpurch"
path_to_right_it = default_path + "rightit"
path_to_wrong = default_path + "wrong"
stop_words = set(stopwords.words('english'))


def read(file):
    text = textract.process(file)
    return text.decode('utf-8')


def extract(cv):
    text = read(cv)
    text = str(text)
    text = text.replace("\n", " ")
    text = text.lower()
    # keyword_dict = pd.read_csv('/home/seemsred/Desktop/Hackathon DS/HackDS/cvproj/cv/media/vac/itmanager.csv')
    return text


def write_file(id, skills, result, resume):
    csv = open("main.csv", "a")
    row = id + "," + skills + "," + str(result) + "," + resume + "\n"
    csv.write(row)


def list_wrong():
    files = [join(path_to_wrong, f) for f in listdir(path_to_wrong) if isfile(join(path_to_wrong, f))]
    i = 0
    # database = pd.DataFrame()
    skills = "none"
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
        words = TreebankWordDetokenizer().detokenize(words)
        # print(dat)
        write_file(str(i), skills, 4, words)


def list_right_sup():
    supfiles = [join(path_to_right_sup, f) for f in listdir(path_to_right_sup) if isfile(join(path_to_right_sup, f))]
    x = 0
    while x < len(supfiles):
        supskills = "mba office logistics english business analysis analytics purchase"
        supfile = supfiles[x]
        supdata = extract(supfile)
        suplang = detect(supdata)
        if suplang == "ru":
            suptrans = TextBlob(supdata)
            supdata = suptrans.translate(from_lang='ru', to='en')
        x += 1
        supdata = str(supdata)
        supdata = supdata.translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.word_tokenize(supdata)
        words = [word for word in tokens if word.isalpha()]
        words = [w for w in words if not w in stop_words]
        words = TreebankWordDetokenizer().detokenize(words)
        write_file(str(x), supskills, 2, words)


def list_right_purch():
    purchfiles = [join(path_to_right_purch, f) for f in listdir(path_to_right_purch) if isfile(join(path_to_right_purch, f))]
    y = 0
    while y < len(purchfiles):
        purchskills = "office ms project aris business analysis analytics purchase procurement"
        purchfile = purchfiles[y]
        purchdata = extract(purchfile)
        purchlang = detect(purchdata)
        if purchlang == "ru":
            purchtrans = TextBlob(purchdata)
            purchdata = purchtrans.translate(from_lang='ru', to='en')
        y += 1
        purchdata = str(purchdata)
        purchdata = purchdata.translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.word_tokenize(purchdata)
        words = [word for word in tokens if word.isalpha()]
        words = [w for w in words if not w in stop_words]
        words = TreebankWordDetokenizer().detokenize(words)
        write_file(str(y), purchskills, 3, words)


def list_right_it():
    itfiles = [join(path_to_right_it, f) for f in listdir(path_to_right_it) if isfile(join(path_to_right_it, f))]
    z = 0
    # database = pd.DataFrame()
    itskills = "office windows exchange active directory itil atc unix linux it"
    while z < len(itfiles):
        itfile = itfiles[z]
        itdat = extract(itfile)
        itlang = detect(itdat)
        if itlang == "ru":
            ittrans = TextBlob(itdat)
            itdat = ittrans.translate(from_lang='ru', to='en')
        z += 1
        itdat = str(itdat)
        itdat = itdat.translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.word_tokenize(itdat)
        words = [word for word in tokens if word.isalpha()]
        words = [w for w in words if not w in stop_words]
        words = TreebankWordDetokenizer().detokenize(words)
        write_file(str(z), itskills, 1, words)


# filename = default_path + "Абдильдин Аян Абубакирович.pdf"
model = []
# df = pd.read_csv("main.csv")
# result = df.result
list_right_it()
print("1")
list_right_sup()
print("2")
list_right_purch()
print("3")
list_wrong()
print("4")

# i = 0
# database = database.append(dat)
# file = read_All_CV(filename)
# office, windows, антивирус, системный администратор, exchange, active directory, itil, atc, домен, unix, linux, it
