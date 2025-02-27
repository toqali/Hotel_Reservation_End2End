from sklearn.ensemble import RandomForestClassifier  # Random Forest

def rf_model():
   model = RandomForestClassifier(n_estimators=200, random_state=42)
   return model
