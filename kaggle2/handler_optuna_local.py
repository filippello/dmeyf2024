import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import optuna
from custom_functions import lgb_gan_eval, objective, ganancia_prob, prepare_df_del_columns, prepare_df_1_clasic
import os

# Cargar configuración desde el archivo YAML
def load_config(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    # Carga la configuración desde config.yaml
    config_path = os.getenv("CONFIG_PATH")
    config = load_config(config_path)
    #config = load_config("/home/fililoco/dmeyf2024/kaggle2/config.yaml")
    #config = load_config("config.yaml")


    
    # Definir rutas y archivos
    if(config["local"]):
        base_path = config['base_path_local']
        dataset_path = config['dataset_path_local']+config["dataset_file"]
        storage_name = "sqlite:///optimization_lgbm.db"
        modelos_path = os.path.join(base_path, config["modelos_path"])
        db_path = os.path.join(base_path, config["db_path"])
        
    else:
        base_path = f'/home/{config["gcloud_user"]}/dmeyf2024/kaggle2'
        dataset_path = f'/home/{config["gcloud_user"]}/buckets/b1/datasets/{config["dataset_file"]}'
        storage_name = f"sqlite:////home/{config["gcloud_user"]}/buckets/b1/db/{config["vm_name"]}/optimization_lgbm.db"
        modelos_path = f'/home/{config["gcloud_user"]}/buckets/b1/models/'
        db_path = f"sqlite:////home/{config["gcloud_user"]}/buckets/b1/datasets/db/"



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

    # Preparar el dataset
    #data = prepare_df_del_columns(data, d_columns)
    if study_data == "only_baja_2":
        X_train, y_train_binaria1, y_train_binaria2, w_train, X_test, y_test_binaria1, y_test_class, w_test = prepare_df_1_clasic(data, mes_train, mes_test)
    else:
        X_train, y_train_binaria1, y_train_binaria2, w_train, X_test, y_test_binaria1, y_test_class, w_test = prepare_df_1_clasic(data, mes_train, mes_test)

        

    # Configurar y correr Optuna
    
    study_name = f"{study_type}_{study_number}_{study_protocol}_data-{study_data}_optuna-{study_number}_timeframe{study_timeframe}_extra-{study_aditional}"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, semillas, X_train, y_train_binaria2, w_train), n_trials=n_trials)
