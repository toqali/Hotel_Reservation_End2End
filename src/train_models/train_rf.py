from src.models_architecture.rf_V1 import rf_model

def create_rf(X_train, y_train):
    model = rf_model()
    model.fit(X_train, y_train)
    return model
