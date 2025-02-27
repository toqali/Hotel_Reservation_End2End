from src.models_architecture.log_V1 import log_model

def create_log(X_train, y_train):
    model = log_model()
    model.fit(X_train, y_train)
    return model
