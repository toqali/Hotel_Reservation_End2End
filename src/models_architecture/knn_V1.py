from sklearn.neighbors import KNeighborsClassifier

def knn_model():
    model = KNeighborsClassifier(n_neighbors=5)
    return model
