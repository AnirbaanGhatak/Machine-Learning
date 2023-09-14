# Name: Anirbaan Ghatak
# Roll no: C026
# Aim: : Implementation of Naïve Bayes Classifier 

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = pd.read_excel("nvb.xlsx")
le = LabelEncoder()

for i in data.columns:
    data[i] = le.fit_transform(data[i])

X = data.drop(columns=['buys_computer'])
y = data['buys_computer']
X_train,X_test,y_train,y_test = train_test_split(X,y)
classifier = GaussianNB()
classifier.fit(X_train,y_train)

accuracy_score(y_test,classifier.predict(X_test))