from src.models_architecture.svm_V1 import svm_model

def create_svm(X_train, y_train):
    model = svm_model()
    model.fit(X_train, y_train)
    return model
