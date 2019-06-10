import numpy as np
import pandas as pd

data_location = "Data/"
train_file = "train.csv"
test_file = "test_id.csv"
sample_submission_file = "sample_submission.csv"

holidays = {"TET_HOLIDAY":["2018-02-11","2018-02-12","2018-02-13","2018-02-14","2018-02-15","2018-02-16","2018-02-17","2018-02-18","2018-02-19","2018-02-20","2018-02-21","2018-02-22","2018-02-23","2019-01-31","2019-02-01","2019-02-02","2019-02-03","2019-02-04","2019-02-05","2019-02-06","2019-02-07","2019-02-08","2019-02-10","2019-02-11","2019-02-12","2019-02-13","2019-02-14","2019-02-15","2020-01-20","2020-01-21","2020-01-22","2020-01-23","2020-01-24","2020-01-25","2020-01-26","2020-01-27","2020-01-28","2020-01-29","2020-01-30"]}
holidays["NORMAL_HOLIDAY"] = ["2018-1-1","2018-04-25","2018-04-30","2018-05-01","2018-09-03","2019-01-01","2019-04-15","2019-04-30","2019-05-01","2019-09-02"]

def data_preprocessing(data, holidays, ZONE_CAT, SERVER_CAT, saved_output="saved_output.csv"):
    """Function to preprocess data
    Input data: data as DataFrame with defined columns (number of rows can be changed)
    Output data: output_data as DataFrame with defined columns and same amount of rows as Input data.
    Preprocessing step:
    - Convert datetime to 3 columns: Year, Month, Day
    - One hot encoding for Year, Month, Day, Hour_ID, Zone_code, Server_Name
    - Add 2 columns: Tet_holiday and Normal_holiday
    - Add column week_date and conver this column to one hot
    """
    temp_data = data.iloc[:,:]
    temp_data["YEAR"]=pd.DatetimeIndex(temp_data["UPDATE_TIME"]).year
    temp_data["MONTH"]=pd.DatetimeIndex(temp_data["UPDATE_TIME"]).month
    temp_data["DATE"]=pd.DatetimeIndex(temp_data["UPDATE_TIME"]).day
    temp_data["WEEK_DATE"]= pd.to_datetime(data["UPDATE_TIME"], errors='coerce').dt.dayofweek
    temp_data["TET_HOLIDAY"]=0
    temp_data["NORMAL_HOLIDAY"]=0
    for index, row in temp_data.iterrows():
        for t1 in holidays["TET_HOLIDAY"]:
            if row["UPDATE_TIME"] == t1:
                row["TET_HOLIDAY"] = 1
        for t2 in holidays["NORMAL_HOLIDAY"]:
            if row["UPDATE_TIME"] == t2:
                row["NORMAL_HOLIDAY"] = 1
    print("tracking")
    dummies = pd.get_dummies(temp_data["ZONE_CODE"])
    dummies = dummies.T.reindex(ZONE_CAT).T.fillna(0)
    temp_data = pd.concat([temp_data, dummies], axis=1)
    dummies = pd.get_dummies(temp_data["SERVER_NAME"])
    dummies = dummies.T.reindex(SERVER_CAT).T.fillna(0)
    temp_data = pd.concat([temp_data, dummies], axis=1)
    temp_data.to_csv(saved_output, index=False)
    return temp_data

train_data = pd.read_csv(data_location+train_file)

ZONE_CAT = train_data["ZONE_CODE"].unique()
SERVER_CAT = train_data["SERVER_NAME"].unique()

data_preprocessing(train_data, holidays, ZONE_CAT, SERVER_CAT, data_location+"preprocessed_train_data.csv")
test_data = pd.read_csv(data_location+test_file)
data_preprocessing(test_data, holidays, ZONE_CAT, SERVER_CAT, data_location+"preprocessed_test_data.csv")