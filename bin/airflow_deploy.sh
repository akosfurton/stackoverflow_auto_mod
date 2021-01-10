# Example bash bin/airflow_deploy.sh subscription_funnel

# Wants 1 parameter dag_name

fname=$(find . | grep -i "dags.*\/$1.py")

# Copy DAG into airflow_user's DAG Bag
echo cp "$fname" /home/airflow_user/airflow/dags/
cp "$fname" /home/airflow_user/airflow/dags/

# Deploy / Update the DAG
echo /home/airflow/user/airflow_user_venv/bin/python /home/airflow-user/airflow/dags/"$1".py
/home/airflow/user/airflow_user_venv/bin/python /home/airflow-user/airflow/dags/"$1".py
