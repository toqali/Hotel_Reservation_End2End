def get_pred_svm(X_test, model):
    y_pred_svm = model.predict(X_test)
    return y_pred_svm