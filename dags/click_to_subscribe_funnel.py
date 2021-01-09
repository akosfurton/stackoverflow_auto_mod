from datetime import datetime

from airflow import DAG
from airflow.operators.bash_operator import BashOperator

from micdrop.utils.constants import RUN_PIPELINE_ENTRY_POINT
from micdrop.utils.send_email import custom_task_failure_alert

email_list = ["akos.furton@gmail.com"]


def task_failure_alert(context):
    dag_id = context.get("task_instance").dag_id
    task_id = context.get("task_instance").task_id
    log_url = context.get("task_instance").log_url

    custom_task_failure_alert(
        email_list=email_list, dag_id=dag_id, task_id=task_id, log_url=log_url
    )


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2020, 5, 1),
    "email": ["none@none.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "on_failure_callback": task_failure_alert,
}

# Will run every sunday at Midnight
dag = DAG("tf_idf_pipeline", schedule_interval="0 0 * * Sun", default_args=default_args)

# Define DAG tasks
task_pre_process = BashOperator(
    task_id="run_preprocessing",
    bash_command=RUN_PIPELINE_ENTRY_POINT,
    params={"module": "pre_processing"},
    dag=dag,
)

task_fit_tf_idf = BashOperator(
    task_id="run_tf_idf",
    bash_command=RUN_PIPELINE_ENTRY_POINT,
    params={"module": "fit_tfidf"},
    dag=dag,
)

task_pre_process >> task_fit_tf_idf
