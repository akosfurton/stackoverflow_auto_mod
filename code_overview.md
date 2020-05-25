# Running the code
 Two setup files are present to automatically install external dependencies
 
 ## Setup.py and requirements.txt
 A requirements.txt file lists external packages and their versions required
  to run the classification pipeline. These packages are automatically
   installed as part of the setup.py file.
   
 - To install the package using pip: `pip install -r requirements.txt` 
 and then `pip install -e .`
 - The package can also be turned into a .whl file using `python setup.py
 bdist_wheel`
 
In addition, there are a number of non-Python dependencies to run the
entire pipeline (setting up data directories, downloading source data). The
provided Dockerfile will automatically set up an environment with all
dependencies set up.

- To create the and run the docker container: 
`docker run -it -p 8888:8888 $(docker build -f Dockerfile_base .)`
 
 
 ## CLI Interface
To run the pipeline, a CLI interface has been created in the
  okcupid_stackoverflow package in the `run.py` script.
  
The CLI interface takes 3 parameters. The parameters are the run_name (used
for distinguishing multiple model runs and providing a unique ID to save
particular models), the module (preprocessing or model fitting), and
use_metadata (which toggles the inclusion of metadata about the post in the
 model fitting step)
 
To run the code from the CLI interface, please use the following syntax from
the top level folder in the repository:
`python okcupid_stackoverflow/run.py --run_name=<INSERT RUN NAME> --module
=<INSERT MODULE> --use_metadata`

## Airflow Orchestration

Airflow is an open source framework to programmatically author and schedule
complex work flows. By chaining individual tasks together, Airflow allows
  users to orchestrate work flows configured as code. Users can view the
   progress of a workflow through Airflow's UI. If a task fails, Airflow also
    provides automated alerting capabilities for the user to resolve and re
    -run the series of tasks.

To run the pipeline in a more automated fashion, an Airflow server has been
 set up at `INSERT IP ADDRESS HERE`. To trigger a particular run of the
  pipeline, please use the entry point found at `bin/airflow_trigger.sh`
  
For example, to trigger a manual run of the tf_idf pipeline: `bash bin
/airflow_trigger.sh -d tf_idf_pipeline -r test_123_must_be_unique -p "--use_metadata"`

This will run the pre-processing task first, and upon completion run the
 model_fitting and model_evaluation tasks. If any of the tasks fail, the
  Airflow server will send an email to the email address listed at the top of
   the DAG definition file (`dags/tf_idf_pipeline.py`)

## Serving Text Predictions with Flask 
- describe Flask for automated serving