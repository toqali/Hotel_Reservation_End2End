from sklearn.metrics import confusion_matrix, accuracy_score

def evaluate_rf(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return cm, acc