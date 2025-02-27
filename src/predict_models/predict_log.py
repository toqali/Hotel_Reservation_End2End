def get_pred_log(X_test, model):
    y_pred_log = model.predict(X_test)
    return y_pred_log