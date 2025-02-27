def get_pred_xgb(X_test, model):
    y_pred_xgb = model.predict(X_test)
    return y_pred_xgb