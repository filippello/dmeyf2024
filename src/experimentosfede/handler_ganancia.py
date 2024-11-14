from custom_functions import lgb_gan_eval, objective, ganancia_prob,prepare_df_1,prepare_df_del_columns,prepare_df_1_clasic,ganancia_prob_iter,ganancia_prob_iter_prom
import yaml

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
import json

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
    else:
        base_path = f'/Users/{config["gcloud_user"]}/Projects/dmeyf2024/src/gcloud'
        dataset_path = f'/Users/{config["gcloud_user"]}/Projects/dmeyf2024/db/{config["dataset_file"]}'

    modelos_path = os.path.join(base_path, config["modelos_path"])
    db_path = os.path.join(base_path, config["db_path"])

    # Leer dataset
    data = pd.read_csv(dataset_path)

    # Extraer variables desde el archivo YAML
    ganancia_acierto = config["ganancia_acierto"]
    costo_estimulo = config["costo_estimulo"]

    #se cambia para el ganancia    
    mes_test_prediccion = config["mes_test_prediccion"]
    mes_train = config["mes_train"]
    #mes_train = config["mes_train"]
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

    #preparamos el dataset
    #borramnos columnas
    data = prepare_df_del_columns(data,d_columns)
    if study_data == "only_baja_2":
        X_train, y_train_binaria1, y_train_binaria2, w_train, X_test, y_test_binaria1, y_test_class, w_test = prepare_df_1_clasic(data, mes_train, mes_test_prediccion)
    elif study_data == "only_baja_3":
        X_train, y_train_binaria1, y_train_binaria2, w_train, X_test, y_test_binaria1, y_test_class, w_test = prepare_df_1(data, mes_train, mes_test_prediccion)

    #b. Creamos el estudio de Optuna
    storage_name = "sqlite:///optimization_lgbm.db"
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

print(X_train.shape)
train_data = lgb.Dataset(X_train,
                          label=y_train_binaria2,
                          weight=w_train)

model_lgb = lgb.train(params,
                  train_data,
                  num_boost_round=best_iter)

model_lgb.save_model(f'./modelos/lgb_{study_type}_{study_number}_{study_data}.txt')

#f. Volvemos a leer el modelo.
model_lgb = lgb.Booster(model_file=modelos_path + f'lgb_{study_type}_{study_number}_{study_data}.txt')


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

""" #ii. Convierto las probabilidades a clases usando el punto de corte 0.025.
clases = [1 if prob >= 0.03 else 0 for prob in predicciones]
#iii. Solo envío estímulo a los registros con probabilidad de "BAJA+2" mayor a 1/40.
X_test['Predicted'] = clases
#v. Selecciono las columnas de interés.
resultados = X_test[["numero_de_cliente", 'Predicted']].reset_index(drop=True)
#vi. Exporto como archivo .csv.
nombre_archivo = "K106_005.csv"

ruta_archivo= f"./exp/{nombre_archivo}"
resultados.to_csv(ruta_archivo, index=False)



#ii. Le pegamos la probabilidad de ser "BAJA" a cada cliente.
X_test['Probabilidad'] = predicciones
#iii. Ordenamos a los clientes por probabilidad de ser "BAJA" de forma descendente.
tb_entrega = X_test.sort_values(by='Probabilidad', ascending=False)
#iv. Genero una lista de distintos cortes candidatos, para enviar a Kaggle.
cortes = range(9000,14000,100)
cortes = [12000]
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
    ruta_archivo= f"./exp/{nombre_archivo}"
    resultados.to_csv(ruta_archivo, index=False)
    
""" """     num_subida_kaggle += 1
    
    #3. Envío a Kaggle.
    #a. Defino los parámetros claves.
    mensaje = f'Archivo {nombre_archivo}. LGB Optuna optimizado con FE (lgb_datos_fe_02_ajusto_optuna), punto_corte: {envios}. No se imputa, 100 Trials para búsqueda de hiperparámetros, 1000 boost con stop early'
    competencia = 'dm-ey-f-2024-primera'
    #c. Subo la Submission.
    api.competition_submit(file_name=ruta_archivo,message=mensaje,competition=competencia) """ """ """