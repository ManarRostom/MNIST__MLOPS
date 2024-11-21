# design an mnist machine learning model

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

digits = load_digits()

X = digits.data
Y = digits.target

model = RandomForestClassifier()

y_pred = model.predict(X)
