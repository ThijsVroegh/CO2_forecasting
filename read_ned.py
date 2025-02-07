from pathlib import Path
from typing import Literal
import pandas

def read_predictions(directory: Path) -> pandas.DataFrame:
    df_sun = read_ned(directory.glob("zon-uur-voorspelling*.csv"), "sun")
    df_wind = read_ned(directory.glob("wind-uur-voorspelling*.csv"), "land-wind")
    df_wind_sea = read_ned(directory.glob("zeewind-uur-voorspelling*.csv"), "sea-wind")

    combined_data = df_sun.copy()
    for data in (df_wind, df_wind_sea):
        combined_data = combined_data.merge(data, on="time", how="outer")

    return combined_data


def read_all(directory: Path) -> pandas.DataFrame:
    # Check if directory has required files
    mix_files = list(directory.glob("electriciteitsmix-*-uur-data.csv"))
    sun_files = list(directory.glob("zon-*-uur-data.csv"))
    wind_files = list(directory.glob("wind-*-uur-data.csv"))
    wind_sea_files = list(directory.glob("zeewind-*-uur-data.csv"))

    if not (mix_files and sun_files and wind_files and wind_sea_files):
        raise FileNotFoundError(
            "Missing required historical data files. "
            "Please run 'python retrieve_ned.py' first to download the data."
        )

    df_mix = read_ned(mix_files, "mix")
    df_sun = read_ned(sun_files, "sun")
    df_wind = read_ned(wind_files, "land-wind")
    df_wind_sea = read_ned(wind_sea_files, "sea-wind")

    combined_data = df_mix.copy()
    for data in (df_sun, df_wind, df_wind_sea):
        combined_data = combined_data.merge(data, on="time", how="outer")

    return combined_data


def read_ned(files, which: Literal["mix", "sun", "land-wind", "sea-wind"]):
    data = []
    for file in sorted(files):
        if which == "mix":
            data.append(read_mix_file(file))
        else:
            data.append(read_production_file(file, which))
    return pandas.concat(data)


def read_mix_file(fname: str | Path):
    df = pandas.read_csv(
        fname,
        usecols=("validfrom (UTC)", "volume (kWh)", "emissionfactor (kg CO2/kWh)"),
    )

    df = df.rename(columns={
        "validfrom (UTC)": "time",
        "volume (kWh)": "total_volume",
        "emissionfactor (kg CO2/kWh)": "emissionfactor",
    })
    df["time"] = pandas.to_datetime(df["time"])
    df = df.set_index("time")
    df["total_volume"] = df["total_volume"].astype(float)
    return df


def read_production_file(fname: str | Path, which: Literal["sun", "land-wind", "sea-wind"]):
    df = pandas.read_csv(
        fname,
        usecols=("validfrom (UTC)", "volume (kWh)"))

    name_vol = f"volume_{which}"
    df = df.rename(columns={
        "validfrom (UTC)": "time",
        "volume (kWh)": name_vol,
    })
    df["time"] = pandas.to_datetime(df["time"])
    df = df.set_index("time")
    df[name_vol] = df[name_vol].astype(float)
    return df
