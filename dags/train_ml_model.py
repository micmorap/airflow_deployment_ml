from airflow.decorators import dag, task
from datetime import datetime

default_args = {
    'owner': 'Michael Mora',
    'start_date': datetime(2024, 10, 8),
    'retries': 1
}

@dag(
    dag_id='train_ml_model_dag_v1',
    default_args = default_args,
    schedule_interval = None,
     catchup = False
     )

def train_ml_model_dag():

    @task.virtualenv(
            task_id="task_venv",
            requirements=[
                "scikit-learn",
                "pandas",
                "seaborn",
                "matplotlib",
                "scipy",
                "keras",
                "numpy",
                "tensorflow",
                "joblib",
                "seaborn"
            ]
    )
    def train_model():
        from sklearn.model_selection import train_test_split
        from joblib import dump
        import pandas as pd
        import numpy as np
        import seaborn as sns
        import os
        import matplotlib.pyplot as plt
        from scipy import stats
        from sklearn.preprocessing import PowerTransformer
        from sklearn.preprocessing import MinMaxScaler
        from keras.models import Sequential
        from keras.layers import LSTM, Dense

        dataset_train_processed = pd.read_csv("/opt/airflow/dags/data/train/processed_training_set_ISA_Historical_Info.csv")
        # dataset_train_processed = pd.read_csv("/Users/michaelandr/Desktop/airflow_deployment_ml/dags/data/train/processed_training_set_ISA_Historical_Info.csv")
        # La red LSTM tendrá como entrada "time_step" datos consecutivos, y como salida 1 dato (la predicción a
        # partir de esos "time_step" datos). Se conformará de esta forma el set de entrenamiento
        time_step = 15
        X_train = []
        Y_train = []
        m = len(dataset_train_processed)

        for i in range(time_step, m):
            # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
            #X_train.append(dataset_processed[i-time_step:i,0])
            X_train.append(dataset_train_processed.iloc[i-time_step:i, 0].values)

            # Y: el siguiente dato
            Y_train.append(dataset_train_processed.iloc[i,0])

        X_train, Y_train = np.array(X_train), np.array(Y_train)     

        # Reshape X_train para que se ajuste al modelo en Keras
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Valores iniciales
        dim_entrada = (X_train.shape[1],1) # 60 datos de una feature
        dim_salida = 1
        na = 50

        # Crear el modelo
        model_lstm = Sequential()

        # Añadir la capa LSTM
        model_lstm.add(LSTM(units=na, return_sequences=True, input_shape= dim_entrada))
        model_lstm.add(LSTM(units=na))

        # Añadir una capa densa para la salida
        model_lstm.add(Dense(dim_salida))

        # Compilar el modelo
        model_lstm.compile(optimizer='rmsprop', loss='mean_squared_error')

        # Resumen del modelo
        model_lstm.summary()

        # Train the model
        model_lstm.fit(X_train, Y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

        # Guarda el modelo en formato HDF5
        #model_lstm.save('/Users/michaelandr/Desktop/airflow_deployment_ml/dags/models/net_lstm.h5')
        model_lstm.save('/opt/airflow/dags/models/model_net_lstm.h5', save_format='h5')

    # Call the task
    train_model()

# Instanciar el DAG
train_ml_model_dag = train_ml_model_dag()