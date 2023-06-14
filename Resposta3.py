from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Create a DecisionTreeClassifier for each criterion
criteria = ["gini", "entropy"]
models = {criterion: DecisionTreeClassifier(criterion=criterion, random_state=42) for criterion in criteria}

# Train each model and evaluate its accuracy
for criterion, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of model with {criterion} criterion: {accuracy:.2f}")
 

# assuming models is your dictionary of trained models
for criterion, model in models.items():
    plt.figure(figsize=(12, 8))
    plot_tree(model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
    plt.title(f"Decision tree using {criterion} criterion")
    plt.show()
