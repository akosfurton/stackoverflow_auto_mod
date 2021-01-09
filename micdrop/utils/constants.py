CITY_MAP = {
    "sfo": "san francisco",
    "kona": "kailua kona",
    "lax": "los angeles",
    "mia": "miami",
    "atl": "atlanta",
    "phx": "phoenix",
    "oak": "oakland",
}

Y_VAR = ["subscriber"]
CATEGORICAL_COLS = ['platform', 'customer_state', 'fav_genre', 'day_of_week', 'city_adj']
NUMERIC_COLS = ['urban_flag', 'credit_card_on_file', 'student', 'is_holiday']

AIRFLOW_SERVER_IP = "x.y.z"

RUN_PIPELINE_ENTRY_POINT = """/home/airflow_user/airflow_user_venv/bin/python \
/home/airflow_user/apple_dataset_challenge/micdrop/run.py \
{{ dag_run.conf.run_id }} \
{{ dag_run.conf.module }} \
{{ dag_run.conf.run_parameters }} """
