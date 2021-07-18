import numpy as np

#compute the euclidean distance between two point of data'
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNeighborsClassifier:

    def __init__(self, num_neighbors=7):
        self.num_neighbors = num_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self.compute_knn(x) for x in X]
        return np.array(y_pred)

    def compute_knn(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[: self.num_neighbors]

        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]

        # return the most common class label
        temp = 0
        what_max = 0
        for data in k_neighbor_labels:
            if (temp < k_neighbor_labels.count(data)):
                temp = k_neighbor_labels.count(data)
                what_max = data
        return what_max