from airflow.decorators import dag, task
from datetime import datetime

default_args = {
    'owner': 'Michael Mora',
    'start_date': datetime(2024, 10, 11),
    'retries': 1
}

@dag(
    dag_id='predictions_lstm_dag_v1',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
)
def run_predictions():
    
    @task.virtualenv(
        task_id="predictions_task_venv",
        requirements=[
            "scikit-learn",
            "pandas",
            "scipy",
            "keras==3.5.0",
            "numpy",
            "tensorflow==2.17.0",
            "joblib"
        ]
    )
    def run_predictions_task():
        import keras
        import pandas as pd

        # Cargar el modelo
        model_file = keras.models.load_model("/opt/airflow/dags/models/net_lstm.h5'")

        # Cargar el dataset
        dataset = pd.read_csv("./data/processed/ISA_Historical_Info_processed.csv")

        # Realizar predicción
        prediction = model_file.predict(dataset)

        return prediction

    # Ejecutar la tarea
    run_predictions_task()

run_prediction = run_predictions()
