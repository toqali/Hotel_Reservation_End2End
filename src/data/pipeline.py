from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from src.data.preprocess import *

def all_pipeline():
    delete_spaces_transformer = FunctionTransformer(delete_frsEnd_spaces)
    handle_date_transformer = FunctionTransformer(handle_date)
    feature_engineering_transformer = FunctionTransformer(feature_engineering)
    log_transformer = FunctionTransformer(log_transform)
    drop_unnecessary_transformer = FunctionTransformer(Drop_unnecessary)
    encode_categorical_transformer = FunctionTransformer(encode_categorical)
    scaling_transformer = FunctionTransformer(scaling_test)
    pipeline = Pipeline([
    ('delete_spaces', delete_spaces_transformer),
    ('handle_date', handle_date_transformer),
    ('feature_engineering', feature_engineering_transformer),
    ('log_transform', log_transformer),
    ('drop_unnecessary', drop_unnecessary_transformer),
    ('encode_categorical', encode_categorical_transformer),
    ('scaling', scaling_transformer)
    ])
    return pipeline