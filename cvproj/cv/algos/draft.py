import numpy as np
import pandas as pd
import re
import nltk
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


nltk.download('stopwords')
airline_tweets = pd.read_csv("main.csv", delimiter=",")

airline_tweets.head()

tags = airline_tweets["result"].values.tolist()

print(tags)

features = airline_tweets["resume"].values.tolist()

processed_features = []

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


vectorizer = TfidfVectorizer (max_features=100, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()


X_train, X_test, y_train, y_test = train_test_split(processed_features, tags, test_size=0.2, random_state=0)


text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)


filename = 'finalized_model.sav'
pickle.dump(text_classifier, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
predictions = loaded_model.predict(X_test)


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))







