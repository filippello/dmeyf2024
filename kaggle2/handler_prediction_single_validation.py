from custom_functions import lgb_gan_eval, objective, ganancia_prob,prepare_df_1,prepare_df_del_columns,prepare_df_1_clasic,ganancia_prob_iter,ganancia_prob_iter_prom
import yaml
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice, plot_contour
from time import time
import pickle
import argparse
import os
# Ejecutar script con los argumentos
# Cargar configuración desde el archivo YAML
def load_config(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    config = load_config("config.yaml")

    # Definir rutas y archivos
    if(config["local"]):
        base_path = config['base_path_local']
        dataset_path = config['dataset_path_local']+config["dataset_file"]
        storage_name = "sqlite:////Users/federicofilippello/Projects/dmeyf2024/kaggle2/db/optimization_lgbm.db"
        modelos_path = os.path.join(base_path, config["modelos_path"])
        db_path = os.path.join(base_path, config["db_path"])
    else:
        base_path = f'/home/{config["gcloud_user"]}/dmeyf2024/kaggle2'
        dataset_path = f'/home/{config["gcloud_user"]}/buckets/b1/datasets/{config["dataset_file"]}'
        storage_name = f"sqlite:////home/{config["gcloud_user"]}/buckets/b1/db/optimization_lgbm.db"
        modelos_path = f'/home/{config["gcloud_user"]}/buckets/b1/models/'
        db_path = f"sqlite:////home/{config["gcloud_user"]}/buckets/b1/datasets/db/"
        exp_path = f'/home/{config["gcloud_user"]}/buckets/b1/exp/'

    #modelos_path = os.path.join(base_path, config["modelos_path"])
    #db_path = os.path.join(base_path, config["db_path"])

    # Leer dataset
    data = pd.read_csv(dataset_path,low_memory=False)

 # Extraer variables desde el archivo YAML
    ganancia_acierto = config["ganancia_acierto"]
    costo_estimulo = config["costo_estimulo"]
    mes_train = config["mes_train"]
    mes_test = config["mes_test"]
    semillas = config["semillas"]
    n_trials = config["n_trials"]
    d_columns = config["d_columns"]

    # Variables relacionadas con el handler
    study_type = config["study_type"]
    study_number = config["study_number"]
    study_data = config["study_data"]
    study_protocol = config["study_protocol"]
    study_timeframe = config["study_timeframe"]
    study_aditional = config["study_aditional"]

    print(f"Usuario de Google Cloud: {config['gcloud_user']}")
    print(f"Archivo de dataset: {dataset_path}")
    print(f"Ruta de modelos: {modelos_path}")
    print(f"Semillas utilizadas: {semillas}")
    print(f"Número de pruebas para Optuna: {n_trials}")
    print(f"Tipo de estudio: {study_type}, Número de estudio: {study_number}")
    print(f"Datos del estudio: {study_data}, Protocolo: {study_protocol}")
    print(f"Marco temporal del estudio: {study_timeframe}, Información adicional: {study_aditional}")
    print(f'storage_name: {storage_name}')

    #preparamos el dataset
    #borramnos columnas
    data = prepare_df_del_columns(data,d_columns)
    #lo fraccionamos
    X_train, y_train_binaria1, y_train_binaria2, w_train, X_test, y_test_binaria1, y_test_class, w_test = prepare_df_1_clasic(data,mes_train,mes_test)

    #b. Creamos el estudio de Optuna
    #storage_name = "sqlite:///optimization_lgbm.db"
    study_name = f"{study_type}_{study_number}_{study_protocol}_data-{study_data}_optuna-{study_number}_timeframe{study_timeframe}_extra-{study_aditional}"
    #cargamos la iter
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )


#c. Tomamos el mejor modelo y con eso entrenamos todos los datos.
best_iter = study.best_trial.user_attrs["best_iter"]
print(f"Mejor cantidad de árboles para el mejor model {best_iter}")
params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'first_metric_only': True,
    'boost_from_average': True,
    'feature_pre_filter': False,
    'max_bin': 31,
    'num_leaves': study.best_trial.params['num_leaves'],
    'learning_rate': study.best_trial.params['learning_rate'],
    'min_data_in_leaf': study.best_trial.params['min_data_in_leaf'],
    'feature_fraction': study.best_trial.params['feature_fraction'],
    'bagging_fraction': study.best_trial.params['bagging_fraction'],
    'seed': semillas[0],
    'verbose': 0
}

train_data = lgb.Dataset(X_train,
                          label=y_train_binaria2,
                          weight=w_train)

model_lgb = lgb.train(params,
                  train_data,
                  num_boost_round=best_iter)


model_file_path = f'{modelos_path}{study_name}.txt'
model_lgb.save_model(model_file_path)

#f. Volvemos a leer el modelo.
model_lgb = lgb.Booster(model_file=model_file_path)


#g. Predecimos Junio.
#i. Realizo la predicción de probabilidades usando el modelo entrenado.
predicciones = model_lgb.predict(X_test)
#ii. Convierto las probabilidades a clases usando el punto de corte 0.025.



#g. Predecimos Junio.
#i. Realizo la predicción de probabilidades usando el modelo entrenado.

predicciones = model_lgb.predict(X_test)
#print(predicciones)
#print(y_train_binaria1.value_counts())

# normal
#gananciaprob = ganancia_prob_iter(predicciones, y_test_binaria1, prop = 1)
#super iter
#gananciaprob , mejor_threshold = ganancia_prob_iter(predicciones, y_test_binaria1, prop = 1)
gananciaprobDF = ganancia_prob_iter_prom(predicciones, y_test_binaria1)
#print("Mejor threshold: ", mejor_threshold)

print("Ganancia: ", gananciaprobDF)

# Leer el archivo si ya tiene datos, o crear una lista vacía si el archivo no existe
# Verificar si el archivo existe y no está vacío
if os.path.exists("ganancia.json") and os.path.getsize("ganancia.json") > 0:
    with open("ganancia.json", "r") as file:
        ganancias_values = json.load(file)
else:
    ganancias_values = []

import csv

if os.path.exists("ganancia.csv") and os.path.getsize("ganancia.csv") > 0:
    ganancias_valuesDF = pd.read_csv("ganancia.csv")
    ganancias_valuesDF = pd.concat([ganancias_valuesDF, pd.DataFrame([gananciaprobDF.values[0]], columns=ganancias_valuesDF.columns)], ignore_index=True)
    print("first")
else:
    print("second")
    ganancias_valuesDF = gananciaprobDF

# Agregar los nuevos valores a la lista existente
#ganancias_values.append(float(gananciaprob))  # Convertir a float si es necesario
#ganancias_values.append((float(gananciaprob),float(mejor_threshold))) 
#ganancias_valuesDF = pd.concat([ganancias_valuesDF, pd.DataFrame([gananciaprobDF.values[0]], columns=ganancias_valuesDF.columns)], ignore_index=True)
# Escribir la lista completa en el archivo
with open("ganancia.json", "w") as file:
    json.dump(ganancias_values, file)

ganancias_valuesDF.to_csv("ganancia.csv", index=False)