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

train_data = lgb.Dataset(X_train,
                          label=y_train_binaria2,
                          weight=w_train)

model_lgb = lgb.train(params,
                  train_data,
                  num_boost_round=best_iter)


model_lgb.save_model('./modelos/lgb_datos_crudos_01.txt')

#f. Volvemos a leer el modelo.
model_lgb = lgb.Booster(model_file=modelos_path + 'lgb_datos_crudos_01.txt')


#g. Predecimos Junio.
#i. Realizo la predicción de probabilidades usando el modelo entrenado.
predicciones = model_lgb.predict(X_test)
#ii. Convierto las probabilidades a clases usando el punto de corte 0.025.
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
    
"""     num_subida_kaggle += 1
    
    #3. Envío a Kaggle.
    #a. Defino los parámetros claves.
    mensaje = f'Archivo {nombre_archivo}. LGB Optuna optimizado con FE (lgb_datos_fe_02_ajusto_optuna), punto_corte: {envios}. No se imputa, 100 Trials para búsqueda de hiperparámetros, 1000 boost con stop early'
    competencia = 'dm-ey-f-2024-primera'
    #c. Subo la Submission.
    api.competition_submit(file_name=ruta_archivo,message=mensaje,competition=competencia) """