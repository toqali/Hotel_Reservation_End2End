import pandas as pd
import os
def load_data():
    # check whether the path found
    data_path = "data/first inten project.csv"
    if not os.path.exists(data_path):
        return f"No such file or directory : {data_path}"
    # read the file
    extension = data_path.split(".")[-1]
    if extension == "csv":
        df = pd.read_csv(data_path)
    else:
        return "Unsupported file format. Provide csv file"
    return df