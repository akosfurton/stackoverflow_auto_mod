import numpy as np
import pandas as pd
from workalendar.usa import UnitedStates

from micdrop.utils.constants import CITY_MAP, FILLNA_DICT, Y_VAR


def load_raw_data(data_path):
    print(f"Reading data from: {data_path}")
    df = pd.read_csv(data_path, encoding="utf-8", parse_dates=["click_date"])
    df.columns = [x.lower() for x in df.columns]
    return df


def fill_null_values(df):
    df = df.fillna(FILLNA_DICT)
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
    df["customer_city"] = df["customer_city"].str.strip().replace(city_map)

    return df


def calc_city_rank(df):
    n_in_city = (
        df.groupby(["customer_city", "customer_state"])
        .size()
        .sort_values(ascending=False)
        .reset_index()
    )
    n_in_city["city_rank_by_size"] = n_in_city.index
    n_in_city["cum_sum"] = n_in_city[0].cumsum()
    n_in_city["total"] = n_in_city[0].sum()
    n_in_city["cum_pct"] = n_in_city["cum_sum"] / n_in_city["total"]

    df = df.merge(
        n_in_city[["customer_city", "city_rank_by_size", "cum_pct"]],
        how="left",
        on="customer_city",
    )
    df["city_adj"] = np.where(df["cum_pct"] < 0.8, df["customer_city"], "other")

    return df


def convert_y_var_to_binary(df, y_var):
    df[y_var] = df[y_var].astype(int)
    return df


def run_preprocessing(base_folder, df=pd.DataFrame(), save_external=False):
    if len(df) == 0:
        df = load_raw_data(f"{base_folder}/data/raw/micdrop_subsciptions_data_v1.csv")

    # Intelligently fill NULL values
    df = fill_null_values(df)

    # Calculate meta-data features
    df = clean_up_cities(df, CITY_MAP)
    df = calc_date_features(df, date_col="click_date")
    df = calc_holidays(df, date_col="click_date")
    df = calc_city_rank(df)

    if Y_VAR in df.columns:
        df = convert_y_var_to_binary(df, Y_VAR)

    if save_external:
        out_path = f"{base_folder}/data/processed/cleaned.parquet"
        print(f"Writing data to: {out_path}")
        df.to_parquet(out_path)

    return df
