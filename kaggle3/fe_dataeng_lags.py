dataset_input = '/home/fililoco/buckets/b1/datasets/competencia_03_pre_sumas.parquet'
import polars as pl
import numpy as np
import yaml
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


competencia_02 = pl.read_parquet(dataset_input)
print(competencia_02.head)
def division_segura(numerador, denominador):
    return pl.when(denominador == 0).then(0).otherwise(numerador / denominador)

df = competencia_02

# Crear las columnas intermedias primero
import polars as pl

# Crear un DataFrame de ejemplo con la columna 'foto_mes' y 'numero_cliente'

# Ordenar por 'numero_cliente' y 'foto_mes' para asegurarse del orden cronológico por cliente
df = df.sort(by=["numero_cliente", "foto_mes"])

# Calcular las métricas por cliente
agg_df = df.groupby("numero_cliente").agg([
    pl.col("value").max().alias("max_1m"),
    pl.col("value").min().alias("min_1m"),
    pl.col("value").mean().alias("avg_1m"),
    pl.col("value").max().alias("max_3m"),
    pl.col("value").min().alias("min_3m"),
    pl.col("value").mean().alias("avg_3m"),
    pl.col("value").max().alias("max_9m"),
    pl.col("value").min().alias("min_9m"),
    pl.col("value").mean().alias("avg_9m")
])

# Unir los resultados agregados con el DataFrame original por 'numero_cliente'
df = df.join(agg_df, on="numero_cliente", how="left")

# Crear lags para 1 mes, 3 meses y 9 meses, pero por cada 'numero_cliente'
df = df.with_columns([
    pl.col("value").shift(1).over("numero_cliente").alias("value_lag_1m"),
    pl.col("value").shift(3).over("numero_cliente").alias("value_lag_3m"),
    pl.col("value").shift(9).over("numero_cliente").alias("value_lag_9m")
])

# Calcular delta entre el primer valor de la ventana y el actual
df = df.with_columns([
    (pl.col("value") - pl.col("value_lag_1m")).alias("delta_1m"),
    (pl.col("value") - pl.col("value_lag_3m")).alias("delta_3m"),
    (pl.col("value") - pl.col("value_lag_9m")).alias("delta_9m")
])

# Mostrar el DataFrame resultante
print(df)



df.write_parquet('/home/fililoco/buckets/b1/datasets/competencia_03_pre_sumas_lags_3_9.parquet')