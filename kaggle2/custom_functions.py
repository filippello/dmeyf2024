import lightgbm as lgb
import numpy as np
import yaml
import numpy as np

# Cargar configuración desde el archivo YAML
def load_config(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config("/home/fililoco/dmeyf2024/kaggle2/config.yaml")
#config = load_config("config.yaml")
semillas = config["semillas"]

def lgb_gan_eval(y_pred, data):
    weight = data.get_weight()
    #ganancia = np.where(weight == 1.00002, config['ganancia_acierto'], 0) - np.where(weight < 1.00002, config['costo_estimulo'], 0)
    ganancia = np.where(weight == 1.00002, config['ganancia_acierto'], 0) - np.where(weight < 1.00002, config['costo_estimulo'], 0)

    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    ganancia = np.cumsum(ganancia)

    return 'gan_eval', np.max(ganancia) , True

def objective(trial, semillas, X_train, y_train_binaria2, w_train): 
    # Rango de parámetros a buscar sus valores óptimos.
    num_leaves = trial.suggest_int('num_leaves', 50, 500)
    learning_rate = trial.suggest_float('learning_rate', 0.005, 0.3) # mas bajo, más iteraciones necesita
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 500, 4000)
    feature_fraction = trial.suggest_float('feature_fraction', 0.5, 1.0)
    bagging_fraction = trial.suggest_float('bagging_fraction', 0.4, 1.0)
    max_depth = trial.suggest_int('max_depth', 3, 15) # Nuevo parámetro.
    lambda_l2 = trial.suggest_float('lambda_l2', 1e-8, 10.0) # Nuevo parámetro.


    # Parámetros que le voy a pasar al modelo.
    params = {
        'objective': 'binary',
        'metric': 'custom',
        'boosting_type': 'gbdt',
        'first_metric_only': True,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'max_bin': 31,
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'min_data_in_leaf': min_data_in_leaf,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'max_depth': max_depth,
        'lambda_l2': lambda_l2,
        'seed': semillas[0],
        'verbose': -1
    }
    
    # Creo el dataset para Light GBM.
    train_data = lgb.Dataset(X_train,
                              label=y_train_binaria2, # eligir la clase
                              weight=w_train)
    
    # Entreno.
    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=1000, # modificar, subit y subir... y descomentar la línea inferior
        callbacks=[lgb.early_stopping(int(50 + 5 / learning_rate))],
        feval=lgb_gan_eval,
        stratified=True,
        nfold=5,
        seed=semillas[0]
    )
    
    # Calculo la ganancia máxima y la mejor iteración donde se obtuvo dicha ganancia.
    max_gan = max(cv_results['valid gan_eval-mean'])
    best_iter = cv_results['valid gan_eval-mean'].index(max_gan) + 1

    # Guardamos cual es la mejor iteración del modelo
    trial.set_user_attr("best_iter", best_iter)

    return max_gan * 5

def ganancia_prob(y_pred, y_true, prop = 1):
  ganancia = np.where(y_true == 1, config['ganancia_acierto'], 0) - np.where(y_true == 0, config['costo_estimulo'], 0)
  return ganancia[y_pred >= 0.0125].sum() / prop

def ganancia_prob_iter(y_pred, y_true, prop = 1):
    ganancia = np.where(y_true == 1, config['ganancia_acierto'], 0) - np.where(y_true == 0, config['costo_estimulo'], 0)

    gananciaMax = -np.inf  
    mejor_threshold = None
    # Utiliza -inf para asegurarte de que cualquier valor será mayor

    # Itera sobre el rango de umbrales
    for threshold in np.arange(0.0125, 0.0125 + 40 * 0.0025, 0.0025):
        gananciaTemp = ganancia[y_pred >= threshold].sum() / prop
        print(gananciaTemp)
        print(threshold)
        gananciaMax = max(gananciaMax, gananciaTemp)
        if gananciaTemp >= gananciaMax:
            gananciaMax = gananciaTemp
            mejor_threshold = threshold

    return gananciaMax, mejor_threshold
import pandas as pd

def ganancia_prob_iter_prom(y_pred, y_true, prop = 1):
    ganancia = np.where(y_true == 1, config['ganancia_acierto'], 0) - np.where(y_true == 0, config['costo_estimulo'], 0)

    # Crear un diccionario para almacenar cada threshold como clave y su ganancia como valor
    resultados = {}

    # Itera sobre el rango de umbrales
    for threshold in np.arange(0.0100, 0.0125 + 40 * 0.0025, 0.0025):
        gananciaTemp = ganancia[y_pred >= threshold].sum() / prop
        resultados[threshold] = gananciaTemp

    # Convertimos el diccionario en un DataFrame con una fila y columnas como los thresholds
    df_resultados = pd.DataFrame([resultados])

    return df_resultados


def prepare_df_del_columns(data,d_columns):
    data = data.drop(d_columns, axis=1)
    return data

def prepare_df_1(data,mes_train,mes_test):
    data['clase_peso'] = 1.0

    data.loc[data['clase_ternaria'] == 'BAJA+3', 'clase_peso'] = 1.00001
    data.loc[data['clase_ternaria'] == 'BAJA+2', 'clase_peso'] = 1.00002
    data.loc[data['clase_ternaria'] == 'BAJA+1', 'clase_peso'] = 1.00001

    data['clase_binaria1'] = 0
    data['clase_binaria2'] = 0
    data['clase_binaria1'] = np.where(data['clase_ternaria'] == 'BAJA+2', 1, 0)
    data['clase_binaria2'] = np.where(data['clase_ternaria'] == 'CONTINUA', 0, 1)
    #solari
    #train_data = data[data['foto_mes'].isin(mes_train)]
    train_data = data[data['foto_mes'].isin(mes_train)]
    test_data = data[data['foto_mes'] == mes_test]

    X_train = train_data.drop(['clase_ternaria', 'clase_peso', 'clase_binaria1','clase_binaria2'], axis=1)
    y_train_binaria1 = train_data['clase_binaria1']
    y_train_binaria2 = train_data['clase_binaria2']
    w_train = train_data['clase_peso']

    X_test = test_data.drop(['clase_ternaria', 'clase_peso', 'clase_binaria1','clase_binaria2'], axis=1)
    y_test_binaria1 = test_data['clase_binaria1']
    y_test_class = test_data['clase_ternaria']
    w_test = test_data['clase_peso']
    return X_train, y_train_binaria1, y_train_binaria2, w_train, X_test, y_test_binaria1, y_test_class, w_test

def prepare_df_1_clasic(data,mes_train,mes_test):
    data['clase_peso'] = 1.0

    data.loc[data['clase_ternaria'] == 'BAJA+2', 'clase_peso'] = 1.00002
    data.loc[data['clase_ternaria'] == 'BAJA+1', 'clase_peso'] = 1.00001

    data['clase_binaria1'] = 0
    data['clase_binaria2'] = 0
    data['clase_binaria1'] = np.where(data['clase_ternaria'] == 'BAJA+2', 1, 0)
    data['clase_binaria2'] = np.where(data['clase_ternaria'] == 'CONTINUA', 0, 1)
    #solari
    #train_data = data[data['foto_mes'].isin(mes_train)]
    train_data = data[data['foto_mes'].isin(mes_train)]
    test_data = data[data['foto_mes'] == mes_test]

    X_train = train_data.drop(['clase_ternaria', 'clase_peso', 'clase_binaria1','clase_binaria2'], axis=1)
    y_train_binaria1 = train_data['clase_binaria1']
    y_train_binaria2 = train_data['clase_binaria2']
    w_train = train_data['clase_peso']

    X_test = test_data.drop(['clase_ternaria', 'clase_peso', 'clase_binaria1','clase_binaria2'], axis=1)
    y_test_binaria1 = test_data['clase_binaria1']
    y_test_class = test_data['clase_ternaria']
    w_test = test_data['clase_peso']
    return X_train, y_train_binaria1, y_train_binaria2, w_train, X_test, y_test_binaria1, y_test_class, w_test


def prepare_df_1_ganancia(data,mes_train,mes_test):
    data['clase_peso'] = 1.0

    data.loc[data['clase_ternaria'] == 'BAJA+3', 'clase_peso'] = 1.00001
    data.loc[data['clase_ternaria'] == 'BAJA+2', 'clase_peso'] = 1.00002
    data.loc[data['clase_ternaria'] == 'BAJA+1', 'clase_peso'] = 1.00001

    data['clase_binaria1'] = 0
    data['clase_binaria2'] = 0
    data['clase_binaria1'] = np.where(data['clase_ternaria'] == 'BAJA+2', 1, 0)
    data['clase_binaria2'] = np.where(data['clase_ternaria'] == 'CONTINUA', 0, 1)
    #solari
    #train_data = data[data['foto_mes'].isin(mes_train)]
    train_data = data[data['foto_mes'].isin(mes_train)]
    test_data = data[data['foto_mes'] == mes_test]

    X_train = train_data.drop(['clase_ternaria', 'clase_peso', 'clase_binaria1','clase_binaria2'], axis=1)
    y_train_binaria1 = train_data['clase_binaria1']
    y_train_binaria2 = train_data['clase_binaria2']
    w_train = train_data['clase_peso']

    X_test = test_data.drop(['clase_ternaria', 'clase_peso', 'clase_binaria1','clase_binaria2'], axis=1)
    y_test_binaria1 = test_data['clase_binaria1']
    y_test_class = test_data['clase_ternaria']
    w_test = test_data['clase_peso']
    return X_train, y_train_binaria1, y_train_binaria2, w_train, X_test, y_test_binaria1, y_test_class, w_test
