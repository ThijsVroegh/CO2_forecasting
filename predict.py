import os

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

import retrieve_ned


def gluonify(df: pd.DataFrame) -> TimeSeriesDataFrame:
    df = df.reset_index()
    df["item_id"] = 0
    return TimeSeriesDataFrame.from_data_frame(df, timestamp_column="time")


if __name__ == "__main__":
    for env_var in ("ned_api_key", "model_path", "csv_path"):
        if os.environ.get(env_var) is None:
            msg = f"Environment variable `{env_var}` not set."
            raise ValueError(msg)

    gluon_runup = gluonify(retrieve_ned.get_runup_data())
    gluon_forecast = gluonify(retrieve_ned.get_current_forecast())

    predictor = TimeSeriesPredictor.load(os.environ.get("model_path"))
    prediction = predictor.predict(gluon_runup, gluon_forecast)
    prediction.to_csv(os.environ.get("csv_path"))
