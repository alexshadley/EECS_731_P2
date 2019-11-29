import re

import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

lines = pd.read_csv('shakespeare-plays/Shakespeare_data.csv')

del lines['Dataline']

# Some of the lines in this dataset are actually stage directions. Since these are not actually spoken by a player, it makes sense that we would want to exclude these points.
lines = lines.query('not ActSceneLine.isnull()')
# Some lines also contain NaN for player?  That's annoying
lines = lines.query('not Player.isnull()')

# Strip out non-alphanumerics
lines = lines.assign(
    PlayerLine=lines['PlayerLine'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x))
)

cv = CountVectorizer()
cv_matrix = cv.fit_transform(lines['PlayerLine'])

le = LabelEncoder()
lines[['ActSceneLine']] = lines[['ActSceneLine']].apply(
    lambda col: le.fit_transform(col)
)

play_onehot = pd.get_dummies(lines['Play'], prefix='Play', sparse=True)

final_data = hstack((lines[['PlayerLinenumber', 'ActSceneLine']], play_onehot, cv_matrix))

X_train, X_test, y_train, y_test = train_test_split(final_data, lines['Player'], test_size=.3, random_state=0)

print('fitting linear')
linear_model = LogisticRegression(max_iter=10)
linear_model.fit(X_train, y_train)

print('scoring linear')
print(linear_model.score(X_test, y_test))

print('fitting bayes')
bayes_model = MultinomialNB()
bayes_model.fit(X_train, y_train)

print('scoring bayes')
print(bayes_model.score(X_test, y_test))

t = tree.DecisionTreeClassifier()
t.fit(X_train, y_train)
print(t.score(X_test, y_test))