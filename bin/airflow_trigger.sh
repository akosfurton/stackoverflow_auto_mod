# Example bash bin/airflow_trigger.sh -d tf_idf_pipeline -r test_123_must_be_unique -p "--use_metadata"

# Wants 3 parameters; these can be given in any order
# -d for dag_name
# -r for run_id (should be unique per run, or per model)
# -p for run_parameters (given as a string with quotes and spaces between parameters)

while getopts "d:r:p:" opt
do
  case "$opt" in
    d) DAG_NAME="$OPTARG";;
    r) RUN_ID="$OPTARG";;
    p) RUN_PARAMETERS="$OPTARG";;
    *) exit 1;;
  esac
done

echo /home/airflow_user/airflow_user_venv/bin/airflow trigger_dag "$DAG_NAME" -r "$RUN_ID" --conf "{\"run_parameter\":\"$RUN_PARAMETERS\"}"
/home/airflow_user/airflow_user_venv/bin/airflow trigger_dag "$DAG_NAME" -r "$RUN_ID" --conf "{\"run_parameter\":\"$RUN_PARAMETERS\"}"