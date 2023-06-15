import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from collections import Counter
from math import log2

iris = load_iris()


# print("Atributos:", iris.feature_names)
# print("Primeira instância:", iris.data[0])


treino_X, teste_X, treino_y, teste_y = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

frequencias = np.array(list(Counter(treino_y).values())) / len(treino_y)

entropia = -np.sum(frequencias * np.log2(frequencias))
print(f"Entropia do conjunto de treinamento: {entropia}")

criterios = ["gini", "entropy","log_loss"]
modelos = {criterio: DecisionTreeClassifier(criterion=criterio, random_state=42) for criterio in criterios}

for criterio, modelo in modelos.items():
    modelo.fit(treino_X, treino_y)
    pred_y = modelo.predict(teste_X)
    acuracia = accuracy_score(teste_y, pred_y)
    print(f"Acurácia do modelo com critério {criterio}: {acuracia:.2f}")

for criterio, modelo in modelos.items():
    plt.figure(figsize=(12, 8))
    plot_tree(modelo, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
    plt.title(f"Árvore de decisão usando critério {criterio}")
    plt.show()

# Referencias
# https://towardsdatascience.com/introduction-to-decision-tree-classifiers-from-scikit-learn-32cd5d23f4d
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

#Calculo - Questão C

# Classe 0 (Setosa): 35 amostras
# Classe 1 (Versicolour): 35 amostras
# Classe 2 (Virginica): 30 amostras

# calcula as entropias: 

# Proporção Classe 0 (Setosa): 35/100 = 0.35
# Proporção Classe 1 (Versicolour): 35/100 = 0.35
# Proporção Classe 2 (Virginica): 30/100 = 0.30

# Agora, substituímos essas proporções na fórmula da entropia:

# Entropia = -∑ (p_i * log2(p_i))
#          = - [0.35*log2(0.35) + 0.35*log2(0.35) + 0.30*log2(0.30)]

# resultado: 

# = -[0.67489+0.67489+0.5]