from src.data import *
from src.train_models import *
from src.predict_models import *
from src.evaluate_models import *
from src.data.pipeline import *

# Step 1: Load the dataset
df = load_data()

# Step 2: Apply preprocessing steps on X_train
X_train, X_test, y_train, y_test = split_data(df)
X_train = delete_frsEnd_spaces(X_train)
X_train = handle_date(X_train)
X_train = feature_engineering(X_train)
X_train = log_transform(X_train)
X_train = Drop_unnecessary(X_train)
X_train = encode_categorical(X_train)
X_train = scaling_train(X_train)

# step 2: Apply the pipeline to your data
pipeline = all_pipeline()
X_test = pipeline.transform(X_test)


# steps 3: Convert categorical vals into 0,1
is_canceled = {
    'Not_Canceled': 0,
    'Canceled': 1
}
y_train = y_train.map(is_canceled)
y_test = y_test.map(is_canceled)

# step 4: fit the models
knn_model = create_knn(X_train, y_train)
'''log_model = create_log(X_train, y_train)
rf_model = create_rf(X_train, y_train)
svm_model = create_svm(X_train, y_train)
xgb_model = create_xgboost(X_train, y_train)'''

# step 5: make predictions
knn_predictions = get_pred_knn(X_test, knn_model)
'''log_predictions = get_pred_log(X_test, log_model)
rf_predictions = get_pred_rf(X_test, rf_model)
svm_predictions = get_pred_svm(X_test, svm_model)
xgb_predictions = get_pred_xgb(X_test, xgb_model)'''

# step 6: Evaluate the models
knn_cm = evaluate_knn(y_test, knn_predictions)
'''log_cm = evaluate_log(y_test, log_predictions)
rf_cm = evaluate_rf(y_test, rf_predictions)
svm_cm = evaluate_svm(y_test, svm_predictions)
xgb_cm = evaluate_xgb(y_test, xgb_predictions)'''

print(knn_cm)