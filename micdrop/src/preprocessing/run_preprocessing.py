import pandas as pd
import numpy as np
from micdrop.utils.constants import CITY_MAP
from workalendar.usa import UnitedStates


def load_raw_data(data_path):
    df = pd.read_csv(data_path, encoding="utf-8", parse_dates=["click_date"])
    df.columns = [x.lower() for x in df.columns]
    return df


def calc_holidays(df, date_col):
    cal = UnitedStates()

    start = df[date_col].min()
    end = df[date_col].max()

    holidays = set(
        holiday[0]
        for year in range(start.year, end.year + 1)
        for holiday in cal.holidays(year)
        if start.date() <= holiday[0] <= end.date()
    )

    df["is_holiday"] = df[date_col].isin(holidays)

    return df


def calc_date_features(df, date_col):
    df["day_of_week"] = df[date_col].dt.day_name()
    return df


def clean_up_cities(df, city_map):
    df["customer_city"] = df["customer_city"].replace(city_map)

    return df

def calc_city_rank(df):
    n_in_city = df.groupby(["customer_city", "customer_state"]).size().sort_values(ascending=False).reset_index()
    n_in_city['city_rank_by_size'] = n_in_city.index
    n_in_city['cum_sum'] = n_in_city[0].cumsum()
    n_in_city['total'] = n_in_city[0].sum()
    n_in_city['cum_pct'] = n_in_city['cum_sum'] / n_in_city['total']

    df = df.merge(n_in_city[['customer_city', 'city_rank_by_size', 'cum_pct']], how="left", on="customer_city")
    df['city_adj'] = np.where(df['cum_pct'] < 0.8, df['customer_city'], "other")

    return df


def run_preprocessing(base_folder, df=pd.DataFrame(), save_external=False):
    if len(df) == 0:
        df = load_raw_data(f"{base_folder}/data/raw/micdrop_subsciptions_data_v1.csv")

    # Calculate meta-data features
    df = clean_up_cities(df, CITY_MAP)
    df = calc_date_features(df, date_col="click_date")
    df = calc_holidays(df, date_col="click_date")

    if save_external:
        df.to_parquet(f"{base_folder}/data/processed/cleaned.parquet")

    return df
