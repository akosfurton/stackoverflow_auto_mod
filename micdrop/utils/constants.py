CITY_MAP = {
    "sfo": "san francisco",
    "kona": "kailua kona",
    "lax": "los angeles",
    "mia": "miami",
    "atl": "atlanta",
    "phx": "phoenix",
    "oak": "oakland",
}

FILLNA_DICT = {
    "customer_city": "unknown",
    "customer_state": "unknown",
    "fav_genre": "unknown",
}

Y_VAR = "subscriber"
CATEGORICAL_COLS = [
    "platform",
    "customer_state",
    "fav_genre",
    "day_of_week",
    "city_adj",
    "urban_flag",
    "credit_card_on_file",
    "student",
    "is_holiday",
]
NUMERIC_COLS = []
EXPECTED_COLUMNS = [
    "click_date",
    "platform",
    "customer_city",
    "customer_state",
    "urban_flag",
    "credit_card_on_file",
    "student",
    "fav_genre",
]

AIRFLOW_SERVER_IP = "x.y.z"

RUN_PIPELINE_ENTRY_POINT = """/home/airflow_user/airflow_user_venv/bin/python \
/home/airflow_user/apple_dataset_challenge/micdrop/run.py \
{{ dag_run.conf.run_id }} \
{{ dag_run.conf.module }} \
{{ dag_run.conf.run_parameters }} """
