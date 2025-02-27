from xgboost import XGBClassifier

def xgb_model():
 model = XGBClassifier(eval_metric='logloss', random_state=42)
 return model