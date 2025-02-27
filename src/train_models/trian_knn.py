from src.models_architecture.knn_V1 import knn_model

def create_knn(X_train, y_train):
    model = knn_model()
    model.fit(X_train, y_train)
    return model
