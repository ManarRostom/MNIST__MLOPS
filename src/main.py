# design an mnist machine learning model

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

digits = load_digits()

X = digits.data
Y = digits.target

model = RandomForestClassifier()

model.fit(X,Y)

y_pred = model.predict(X)
accuracy = accuracy_score(Y,y_pred)