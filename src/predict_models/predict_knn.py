

def get_pred_knn(X_test, model):
    y_pred_knn = model.predict(X_test)
    return y_pred_knn