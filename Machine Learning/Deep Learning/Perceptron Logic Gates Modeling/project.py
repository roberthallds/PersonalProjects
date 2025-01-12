from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# 1. define AND logic gate dataset
data = [[0, 0], [1, 0], [0, 1], [1, 1]]

# 2. define AND logic gate labels
labels_AND = [0, 0, 0, 1]

# 3. plot labels against data
plt.scatter([point[0] for point in data], [point[1] for point in data], c=labels_AND)
plt.xlabel("AND Gates")
plt.ylabel("Values")
plt.show()
plt.clf()

# 4. build perceptron object with 
#    40 max iterations and 22 random state
classifier_AND = Perceptron(max_iter=40, random_state=22)

# 5. train model on data and labels
classifier_AND.fit(data, labels_AND)

# 6. score classifier, ensuring perfect accuracy
print(f"AND Accuracy: {classifier_AND.score(data, labels_AND)}")

# 7. create and train new perceptron on XOR
labels_XOR = [0, 1, 1, 0]
clf_XOR = Perceptron(max_iter=40, random_state=22)
clf_XOR.fit(data, labels_XOR)
print(f"XOR Accuracy: {clf_XOR.score(data, labels_XOR)}")

# 8. create and train new perceptron on OR
labels_OR = [0, 1, 1, 1]
clf_OR = Perceptron(max_iter=40, random_state=22)
clf_OR.fit(data, labels_OR)
print(f"OR Accuracy: {clf_OR.score(data, labels_OR)}")

# 9. experiment with .decision_function() on 
#    given parameter matrix using the AND
#    classifier
print(f"Decision on Sample Parameter Matrix: {classifier_AND.decision_function([[0, 0], [1, 1], [0.5, 0.5]])}")

# 10-16. create heatmap representing the
#        decision boundary
x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)
point_grid = list(product(x_values, y_values))
distances = classifier_AND.decision_function(point_grid)
abs_distances = abs(distances)
distances_matrix = np.reshape(abs_distances, [100, 100])
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)
plt.colorbar(heatmap)
plt.show()
plt.clf()