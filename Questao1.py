from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import numpy as np

dados = load_breast_cancer()

intervalo_suavizacao = np.linspace(1e-11, 1e-8, 100)

melhor_pontuacao = 0
melhor_suavizacao = 0

for suavizacao in intervalo_suavizacao:
    classificador = GaussianNB(var_smoothing=suavizacao)
    pontuacoes = cross_val_score(classificador, dados.data, dados.target, cv=5)
    media_pontuacao = np.mean(pontuacoes)
    if media_pontuacao > melhor_pontuacao:
        melhor_pontuacao = media_pontuacao
        melhor_suavizacao = suavizacao

print(f'O melhor valor para suavizacao é {melhor_suavizacao} com uma acurácia média de {melhor_pontuacao}.')
