NOT_METADATA_COLS = [
    "body",
    "title",
    "label",
    "light_cleaned_title",
    "light_cleaned_body",
    "light_cleaned_text",
    "cleaned_title",
    "cleaned_body",
    "cleaned_text",
]

AIRFLOW_SERVER_IP = "x.y.z"

RUN_PIPELINE_ENTRY_POINT = """/home/airflow_user/airflow_user_venv/bin/python \
/home/airflow_user/okcupid/okcupid_stackoverflow/run.py \
{{ dag_run.conf.run_id }} \
{{ dag_run.conf.module }} \
{{ dag_run.conf.run_parameters }} """
