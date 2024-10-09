from custom_functions import lgb_gan_eval, objective, ganancia_prob,prepare_df_1,prepare_df_del_columns


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

# Definir argumentos para el script
def parse_args():
    parser = argparse.ArgumentParser(description="Procesar variables desde línea de comando")
    
    # Argumentos obligatorios
    parser.add_argument('--gcloud_user', type=str, required=True, help='Nombre del usuario de Google Cloud')
    parser.add_argument('--dataset_file', type=str, required=True, help='Archivo del dataset')
    parser.add_argument('--ganancia_acierto', type=int, required=True, help='Ganancia por acierto')
    parser.add_argument('--costo_estimulo', type=int, required=True, help='Costo de estímulo')
    parser.add_argument('--mes_train', type=int, required=True, help='Mes de entrenamiento')
    parser.add_argument('--mes_test', type=int, required=True, help='Mes de prueba')
    parser.add_argument('--modelos_path', type=str, required=True, help='Ruta de los modelos')
    parser.add_argument('--db_path', type=str, required=True, help='Ruta de la base de datos')
    parser.add_argument('--semillas', type=int, nargs='+', required=True, help='Lista de semillas')
    parser.add_argument('--n_trials', type=int, required=True, help='Número de pruebas de Optuna')
    parser.add_argument('--d_columns', type=str, nargs='+',required=True, help='columnas a borrar')

    # Variables adicionales relacionadas con el handler
    parser.add_argument('--study_type', type=str, required=True, help='Tipo de estudio')
    parser.add_argument('--study_number', type=str, required=True, help='Número de estudio')
    parser.add_argument('--study_data', type=str, required=True, help='Datos del estudio')
    parser.add_argument('--study_protocol', type=str, required=True, help='Protocolo del estudio (e.g., lgmb)')
    parser.add_argument('--study_timeframe', type=str, required=True, help='Marco temporal del estudio')
    parser.add_argument('--study_aditional', type=str, required=True, help='Información adicional del estudio')

    return parser.parse_args()

# Ejecutar script con los argumentos
if __name__ == "__main__":
    args = parse_args()

    # Rutas y archivos
    #base_path = f'/home/{args.gcloud_user}/buckets/b1/public/gcloud/' 
    #dataset_path =  f'/home/{args.gcloud_user}/buckets/b1/datasets/{args.dataset_file}'
    base_path = f'/Users/federicofilippello/Projects/dmeyf2024/src/gcloud' 
    dataset_path =  f'/Users/federicofilippello/Projects/dmeyf2024/db/{args.dataset_file}'
    modelos_path = os.path.join(base_path, args.modelos_path)
    db_path = f'base_path+args.db_path'
    
    # Leer dataset
    data = pd.read_csv(dataset_path)

    # Variables adicionales
    ganancia_acierto = args.ganancia_acierto
    costo_estimulo = args.costo_estimulo
    mes_train = args.mes_train
    mes_test = args.mes_test
    semillas = args.semillas
    n_trials = args.n_trials
    d_columns = args.d_columns

    # Variables relacionadas con el handler
    study_type = args.study_type
    study_number = args.study_number
    study_data = args.study_data
    study_protocol = args.study_protocol
    study_timeframe = args.study_timeframe
    study_aditional = args.study_aditional

    print(f"Usuario de Google Cloud: {args.gcloud_user}")
    print(f"Archivo de dataset: {dataset_path}")
    print(f"Ruta de modelos: {modelos_path}")
    print(f"Semillas utilizadas: {semillas}")
    print(f"Número de pruebas para Optuna: {n_trials}")
    print(f"Tipo de estudio: {study_type}, Número de estudio: {study_number}")
    print(f"Datos del estudio: {study_data}, Protocolo: {study_protocol}")
    print(f"Marco temporal del estudio: {study_timeframe}, Información adicional: {study_aditional}")
    
#preparamos el dataset
#borramnos columnas
data = prepare_df_del_columns(data,d_columns)
#lo fraccionamos
X_train, y_train_binaria1, y_train_binaria2, w_train, X_test, y_test_binaria1, y_test_class, w_test = prepare_df_1(data,mes_train,mes_test)


    # Aquí continúa el código para entrenar el modelo, realizar el tuning con Optuna, etc.
#a. Voy a realizar un estudio de Optuna para encontrar los mejores parámetros.
#i. Creo la base de datos donde guardar los resultados.
storage_name = "sqlite:///optimization_lgbm.db"
study_name = f"{study_type}_{study_number}_{study_protocol}_data-{study_data}_optuna-{study_number}_timeframe{study_timeframe}_extra-{study_aditional}"
#study_name = "exp_300_lgbm_datos_crudos_100_num_boost_round"

#ii. Creo el estudio.
study = optuna.create_study(
    direction="maximize",
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True,
)

#iii. Corro el estudio.
#study.optimize(objective, n_trials)
study.optimize(lambda trial: objective(trial, semillas,X_train,y_train_binaria2,w_train), n_trials=n_trials)