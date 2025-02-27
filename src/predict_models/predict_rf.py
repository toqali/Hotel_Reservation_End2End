def get_pred_rf(X_test, model):
    y_pred_rf = model.predict(X_test)
    return y_pred_rf