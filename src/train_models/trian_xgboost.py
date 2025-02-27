from src.models_architecture.xgboost_V1 import xgb_model

def create_xgboost(X_train, y_train):
    model = xgb_model()
    model.fit(X_train, y_train)
    return model
