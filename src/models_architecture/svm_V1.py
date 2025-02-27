from sklearn.svm import SVC

def svm_model():
   model = SVC(random_state=42)
   return model