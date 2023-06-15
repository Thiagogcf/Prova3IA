from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import numpy as np

data = load_breast_cancer()

var_smoothing_range = np.linspace(1e-11, 1e-8, 100)

best_score = 0
best_var_smoothing = 0

for var_smoothing in var_smoothing_range:
    classifier = GaussianNB(var_smoothing=var_smoothing)
    scores = cross_val_score(classifier, data.data, data.target, cv=5)
    mean_score = np.mean(scores)
    if mean_score > best_score:
        best_score = mean_score
        best_var_smoothing = var_smoothing

print(f'O melhor valor para var_smoothing é {best_var_smoothing} com uma acurácia média de {best_score}.')
