import datetime
import json
import os
from typing import Literal
import pandas
import requests

from config import FORECAST_DIR, HISTORICAL_DIR, DOWNLOADED_DIR


def get_key() -> str:
    key = os.environ.get("ned_api_key")
    if key is None:
        msg = "`ned_api_key` not found in environment variables."
        raise ValueError(msg)
    return key


TYPE_CODES = {
    "sun": 2,
    "sea-wind": 17,
    "land-wind": 1,
    "mix": 27,
}

URL = "https://api.ned.nl/v1/utilizations"

HEADERS = {
    "X-AUTH-TOKEN": get_key(),
    "accept": "application/ld+json",
}

DATE_FORMAT = "%Y-%m-%d"
RUNUP_PERIOD = 4*7 # 4 weeks


def get_last_page(response: requests.Response) -> int:
    """Retrieve the last page nr, as data can be split over multiple pages."""
    json_dict = json.loads(response.text)
    return int(json_dict['hydra:view']['hydra:last'].split("&page=")[-1])


def request_data(
    start_date: str,
    end_date: str,
    forecast: bool,
    which: Literal["mix", "sun", "sea-wind", "land-wind"],
    page: int = 1,
):
    params = {
        "point": 0,  # NL
        "type": TYPE_CODES[which],
        "granularity": 5,  # Hourly
        "granularitytimezone": 0,  # UTC
        "classification": 1 if forecast else 2,  # historic=2, forecast=1
        "activity": 1,  # Providing
        "validfrom[after]": start_date,  # from (including)
        "validfrom[strictly_before]": end_date,  # up to (excluding)
        "page": page,
    }
    response = requests.get(URL, headers=HEADERS, params=params, allow_redirects=False)
    if response.status_code != 200:
        msg = f"Error retrieving data from api.ned.nl. Status code {response.status_code}"
        raise requests.ConnectionError(msg)
    return response


def parse_response(
    response: requests.Response, which: Literal["mix", "sun", "land-wind", "sea-wind"]
):
    json_dict = json.loads(response.text)
    if which == "mix":
        vol = list()
        ef = list()
        dtime = list()
        for el in json_dict['hydra:member']:
            vol.append(float(el["volume"]))
            ef.append(el["emissionfactor"])
            dtime.append(pandas.Timestamp(el["validfrom"]))
        df = pandas.DataFrame(
            data={"total_volume": vol, "emissionfactor": ef},
            index=dtime
        )
    else:
        vol = list()
        dtime = list()
        for el in json_dict['hydra:member']:
            vol.append(float(el["volume"]))
            dtime.append(pandas.Timestamp(el["validfrom"]))
        df = pandas.DataFrame(data={f"volume_{which}": vol}, index=dtime)

    df.index = df.index.strftime("%Y-%m-%d %H:%M:%S")
    df.index.name = "time"
    return df


def get_data(
    sources: tuple[str], start_date: str, end_date: str, forecast: bool
) -> pandas.DataFrame:
    dfs = {source: [] for source in sources}
    for source in sources:
        response = request_data(
            start_date,
            end_date,
            forecast=forecast,
            which=source,
            page=1,
        )
        dfs[source].append(parse_response(response, source))

        # Requests >200 items will have multiple pages. Retrieve and append these.
        last_page = get_last_page(response)
        if last_page >=2 :
            for page in range(2, last_page + 1):
                response = request_data(
                    start_date,
                    end_date,
                    forecast=forecast,
                    which=source,
                    page=page,
                )
                dfs[source].append(parse_response(response, source))
    return pandas.concat(
        [pandas.concat(page, axis=0) for page in dfs.values()],
        axis=1,
    )


def get_current_forecast() -> pandas.DataFrame:
    now = datetime.datetime.now()
    start_forecast = now.strftime(DATE_FORMAT)
    end_forecast = (now + datetime.timedelta(days=7)).strftime(DATE_FORMAT)

    sources = ("sun", "land-wind", "sea-wind")
    df = get_data(sources, start_forecast, end_forecast, forecast=True)
    
    # Save forecast data
    timestamp = now.strftime('%Y%m%d')  # Consistent timestamp format
    forecast_file = FORECAST_DIR / f"forecast_{timestamp}.csv"
    downloaded_file = DOWNLOADED_DIR / f"forecast_{timestamp}.csv"
    df.to_csv(forecast_file)
    df.to_csv(downloaded_file)
    return df


def get_runup_data() -> pandas.DataFrame:
    """Download recent historical data for model context.
    
    Downloads the last 4 weeks of historical data (defined by RUNUP_PERIOD) to provide
    recent context for the forecasting model. This includes:
    - Electricity mix data (total volume and emission factor)
    - Solar production
    - Onshore wind production
    - Offshore wind production
    
    The data is saved in two locations:
    1. HISTORICAL_DIR/historical_{date}.csv
    2. DOWNLOADED_DIR/historical_{date}.csv
    
    Both files contain the same combined data with all sources.
    
    Returns:
        pandas.DataFrame: Combined dataframe containing recent historical data with columns:
            - total_volume: Total electricity volume
            - emissionfactor: CO2 emission factor
            - volume_sun: Solar production
            - volume_land-wind: Onshore wind production
            - volume_sea-wind: Offshore wind production
    
    Raises:
        requests.ConnectionError: If the API request fails
        ValueError: If the API key is not found in environment variables
    
    Note:
        The runup period is defined by RUNUP_PERIOD constant (default: 4 weeks)
        This data is used to provide recent context for the forecasting model.
    """
    now = datetime.datetime.now()
    start_runup = (now - datetime.timedelta(days=RUNUP_PERIOD)).strftime(DATE_FORMAT)
    end_runup = now.strftime(DATE_FORMAT)

    sources = ("mix", "sun", "land-wind", "sea-wind")
    df = get_data(sources, start_runup, end_runup, forecast=False)
    
    # Save historical data
    timestamp = now.strftime('%Y%m%d')  # Consistent timestamp format
    historical_file = HISTORICAL_DIR / f"historical_{timestamp}.csv"
    downloaded_file = DOWNLOADED_DIR / f"historical_{timestamp}.csv"
    df.to_csv(historical_file)
    df.to_csv(downloaded_file)
    return df


def get_historical_data() -> pandas.DataFrame:
    """Download and save historical data for model training.
    
    Downloads 5 years of historical data from the NED API, including:
    - Electricity mix data (total volume and emission factor)
    - Solar production
    - Onshore wind production
    - Offshore wind production
    
    The data is saved in two formats:
    1. A combined CSV file containing all data columns, saved to DOWNLOADED_DIR
    2. Separate CSV files for each source with standardized column names, saved to HISTORICAL_DIR:
        - electriciteitsmix-{date}-uur-data.csv: Contains total volume and emission factor
        - zon-{date}-uur-data.csv: Solar production
        - wind-{date}-uur-data.csv: Onshore wind production
        - zeewind-{date}-uur-data.csv: Offshore wind production
    
    Returns:
        pandas.DataFrame: Combined dataframe containing all downloaded data with columns:
            - total_volume: Total electricity volume
            - emissionfactor: CO2 emission factor
            - volume_sun: Solar production
            - volume_land-wind: Onshore wind production
            - volume_sea-wind: Offshore wind production
    
    Raises:
        requests.ConnectionError: If the API request fails
        ValueError: If the API key is not found in environment variables
    """
    now = datetime.datetime.now()
    # Get 4 years of historical data
    start_date = (now - datetime.timedelta(days=4*365)).strftime(DATE_FORMAT)
    end_date = now.strftime(DATE_FORMAT)

    print(f"Downloading data from {start_date} to {end_date}...")
    sources = ("mix", "sun", "land-wind", "sea-wind")
    df = get_data(sources, start_date, end_date, forecast=False)
    
    # Save combined data
    timestamp = now.strftime('%Y%m%d')
    combined_file = DOWNLOADED_DIR / f"historical_combined_{timestamp}.csv"
    df.to_csv(combined_file)
    print(f"Saved combined historical data to {combined_file}")
    
    # Save mix data with correct column names
    if "total_volume" in df.columns:
        mix_df = df[["total_volume", "emissionfactor"]].copy()
        mix_df.columns = ["volume (kWh)", "emissionfactor (kg CO2/kWh)"]
        mix_df.index.name = "validfrom (UTC)"
        mix_df.to_csv(
            HISTORICAL_DIR / f"electriciteitsmix-{timestamp}-uur-data.csv"
        )
    
    # Save source-specific data with correct column names
    source_names = {
        "sun": "zon",
        "land-wind": "wind",
        "sea-wind": "zeewind"
    }
    
    for source, name in source_names.items():
        col = f"volume_{source}"
        if col in df.columns:
            source_df = df[[col]].copy()
            source_df.columns = ["volume (kWh)"]
            source_df.index.name = "validfrom (UTC)"
            source_df.to_csv(
                HISTORICAL_DIR / f"{name}-{timestamp}-uur-data.csv"
            )
    
    print(f"Saved split historical data files to {HISTORICAL_DIR}")
    return df
