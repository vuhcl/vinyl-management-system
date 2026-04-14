from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def print_hello():
    print("Hello from Airflow!")
    return "Hello task completed"


with DAG(
    'example_dag',
    default_args=default_args,
    description='A simple example DAG',
    schedule=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['example'],
) as dag:

    task_hello = PythonOperator(
        task_id='hello_task',
        python_callable=print_hello,
    )

    task_date = BashOperator(
        task_id='print_date',
        bash_command='date',
    )

    task_hello >> task_date
