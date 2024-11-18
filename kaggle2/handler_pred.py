from custom_functions import lgb_gan_eval, objective, ganancia_prob,prepare_df_del_columns,prepare_df_1_clasic,ganancia_prob_iter,ganancia_prob_iter_prom,prepare_df_pred
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
    #config = load_config("config.yaml")
    config_path = os.getenv("CONFIG_PATH")
    config = load_config(config_path)

    # Definir rutas y archivos
    if(config["local"]):
        base_path = config['base_path_local']
        dataset_path = "/Users/federicofilippello/Projects/dmeyf2024/kaggle2/datasets_competencia_02_DE_1y.csv"
        storage_name = "sqlite:////Users/federicofilippello/Projects/dmeyf2024/kaggle2/db/optimization_lgbm.db"
        modelos_path = os.path.join(base_path, config["modelos_path"])
        db_path = os.path.join(base_path, config["db_path"])
        entregas_path = '/Users/federicofilippello/Projects/dmeyf2024/kaggle2/entregas/'
    else:
        base_path = f'/home/{config["gcloud_user"]}/dmeyf2024/kaggle2'
        dataset_path = f'/home/{config["gcloud_user"]}/buckets/b1/datasets/{config["dataset_file"]}'
        storage_name = f"sqlite:////home/{config["gcloud_user"]}/buckets/b1/db/optimization_lgbm.db"
        modelos_path = f'/home/{config["gcloud_user"]}/buckets/b1/models/'
        db_path = f"sqlite:////home/{config["gcloud_user"]}/buckets/b1/datasets/db/"
        exp_path = f'/home/{config["gcloud_user"]}/buckets/b1/exp/'
        entregas_path = f'/home/{config["gcloud_user"]}/buckets/b1/entregas/'

    #modelos_path = os.path.join(base_path, config["modelos_path"])
    #db_path = os.path.join(base_path, config["db_path"])

    # Leer dataset
    data = pd.read_csv(dataset_path,low_memory=False)

 # Extraer variables desde el archivo YAML
    ganancia_acierto = config["ganancia_acierto"]
    costo_estimulo = config["costo_estimulo"]
    mes_train = config["mes_train"]
    mes_test = config["mes_test"]
    mes_pred = config["mes_test_prediccion"]
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
    X_pred = prepare_df_pred(data,mes_pred)
    #b. Creamos el estudio de Optuna
    #storage_name = "sqlite:///optimization_lgbm.db"
    study_name = f"{study_type}_{study_number}_{study_protocol}_data-{study_data}_optuna-{study_number}_timeframe{study_timeframe}_extra-{study_aditional}"
    #cargamos la iter

model_file_path = f'{modelos_path}{study_name}.txt'

#f. Volvemos a leer el modelo.
model_lgb = lgb.Booster(model_file=model_file_path)


#g. Predecimos Junio.
#i. Realizo la predicción de probabilidades usando el modelo entrenado.
#predicciones = model_lgb.predict(X_test)
#ii. Convierto las probabilidades a clases usando el punto de corte 0.025.



#g. Predecimos Junio.
#i. Realizo la predicción de probabilidades usando el modelo entrenado.

predicciones = model_lgb.predict(X_pred)


#ii. Le pegamos la probabilidad de ser "BAJA" a cada cliente.
X_pred['Probabilidad'] = predicciones
#iii. Ordenamos a los clientes por probabilidad de ser "BAJA" de forma descendente.
tb_entrega = X_pred.sort_values(by='Probabilidad', ascending=False)
#iv. Genero una lista de distintos cortes candidatos, para enviar a Kaggle.
#cortes = range(9500,12500,100)
cortes = [10600]
#v. Generamos las distintas predicciones de clases a partir de los distintos cortes posibles.
num_subida_kaggle = 1

for envios in cortes:
    #1. Le ponemos clase 1 ("BAJA") a los primeros "envios" con mayor probabilidad.
    tb_entrega['Predicted'] = 0
    tb_entrega.iloc[:envios, tb_entrega.columns.get_loc('Predicted')] = 1
    resultados = tb_entrega[["numero_de_cliente", 'Predicted']].reset_index(drop=True)
    
    print("Cantidad de clientes {}".format(envios))
    #2. Guardamos el archivo para Kaggle.
    nombre_archivo = "K107_00{}.csv".format(num_subida_kaggle)
    ruta_archivo= f"{entregas_path}{nombre_archivo}"
    resultados.to_csv(ruta_archivo, index=False)






