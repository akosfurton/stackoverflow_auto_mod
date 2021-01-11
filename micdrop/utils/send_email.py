# Make sure that the airflow.cfg file in ~/airflow/airflow.cfg has the [SMTP]
# settings set
# smtp_host = SMTP server IP address
# smtp_mail_from = from email address (airflow@micdrop.com)

from airflow.utils.email import send_email_smtp

from micdrop.utils.constants import AIRFLOW_SERVER_IP


def custom_task_failure_alert(email_list, dag_id, task_id, log_url):
    subject = (
        f"[Airflow] DAG {dag_id.capitalize()} - Task {task_id.capitalize()}: Failed"
    )

    html_content = f"""
    Hello,<br>
    <br>
    The <b>{task_id.capitalize()}</b> task
    of the <b>{dag_id.capitalize()}</b> DAG has failed.<br>
    <br>
    Please visit the Airflow log page at
    {log_url.replace("localhost", AIRFLOW_SERVER_IP)} for more information.<br>
    <br>
    Cheers,<br>
    Airflow log bot
    """

    send_email_smtp(email_list, subject, html_content)
