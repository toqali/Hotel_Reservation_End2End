import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


scaler = joblib.load(r"src\saved_preprocessings\scaler_hotel_v1.pkl")

def split_data(df):
    x = df.drop("booking status", axis = 1)
    y = df["booking status"]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def scaling_train(X_train):
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(scaled_array, columns=X_train.columns)
    return X_train

def scaling_test(X_test):
    scaled_array = scaler.transform(X_test)
    X_test = pd.DataFrame(scaled_array, columns=X_test.columns)
    return X_test

def delete_frsEnd_spaces(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df

def handle_date(df):
    df=df.copy()
    df['date of reservation'] = pd.to_datetime(df['date of reservation'], errors = 'coerce')
    df['date of reservation'] = df['date of reservation'].fillna("3/1/2018")
    return df
def feature_engineering(df):
    df = df.copy()
    df['year'] = df['date of reservation'].dt.year
    df['month'] = df['date of reservation'].dt.month
    df['day'] = df['date of reservation'].dt.day
    df['total_stay_night'] = df['number of week nights'] + df['number of weekend nights']
    df['total_visiors'] = df['number of adults'] + df['number of children']
    df['percent_canceled'] = df['P-C'] / (df['P-C'] + df['P-not-C'])
    df['percent_canceled'] = df['percent_canceled'].fillna(-1)
    return df

def encode_categorical(df):
    df = df.copy()
    meal_dict = {
        "Not Selected": 1, 
        'Meal Plan 1': 2, 
        'Meal Plan 2': 3, 
        'Meal Plan 3': 4}
    room_dict = {
        'Room_Type 1': 1, 
        'Room_Type 2': 2, 
        'Room_Type 3': 3, 
        'Room_Type 4': 4, 
        'Room_Type 5': 5, 
        'Room_Type 6': 6, 
        'Room_Type 7': 7
    }
    seg_dict = {"Online": 5, 'Offline': 4, 'Corporate': 3, 'Complementary': 1, "Aviation":2}


    df['type of meal'] = df['type of meal'].map(meal_dict)
    df['room type'] = df['room type'].map(room_dict)
    df['market segment type'] = df['market segment type'].map(seg_dict)

    return df

def Drop_unnecessary(df):
    df = df.copy()
    df = df.drop(columns=[
        "Booking_ID", 
        "date of reservation", 
        "number of adults",
        "number of children",
        "number of weekend nights",
        "number of week nights",
        "repeated"])
    return df

def log_transform(df):
    df = df.copy()
    print(type(df))
    df[['lead time', 'average price']] = df[['lead time', 'average price']].astype(float)  # Convert first
    df[['lead time', 'average price']] = np.log1p(df[['lead time', 'average price']])  # Apply log
    return df
