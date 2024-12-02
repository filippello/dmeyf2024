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

    # Definir rutas y archivos
    if(config["local"]):
        base_path = config['base_path_local']
        dataset_path = config['dataset_path_local'] + config["dataset_file"]
        storage_name = "sqlite:///optimization_lgbm.db"
        modelos_path = os.path.join(base_path, config["modelos_path"])
        db_path = os.path.join(base_path, config["db_path"])
    else:
        base_path = f'/home/{config["gcloud_user"]}/dmeyf2024/kaggle2'
        dataset_path = f'/home/{config["gcloud_user"]}/buckets/b1/datasets/{config["dataset_file"]}'
        storage_name = f"sqlite:////home/{config['gcloud_user']}/buckets/b1/db/{config['vm_name']}/optimization_lgbm.db"
        modelos_path = f'/home/{config["gcloud_user"]}/buckets/b1/models/'
        db_path = f"sqlite:////home/{config['gcloud_user']}/buckets/b1/datasets/db/"


    foto_mes_excluir = config['mes_test_prediccion']
    # Cargar el dataset
    dataset_input = '/home/fililoco/buckets/b1/datasets/competencia_03_pre_sumas_lags_3_9.parquet'
    competencia_02 = pl.read_parquet(dataset_input)
    print(competencia_02.head())
    df = competencia_02
    meses_totales = config['mes_train'].append(config['mes_test_prediccion'])
  
    # Filtramos los registros donde 'clase_ternaria' sea igual a 'CONTINUA'
    df_continua = df.filter(pl.col('clase_ternaria') == 'CONTINUA')

    # Excluimos el 'foto_mes' específico
    df_continua_excluido = df_continua.filter(pl.col('foto_mes') != foto_mes_excluir)

    # Agrupamos por 'foto_mes' y aplicamos un undersampling del 20% para cada grupo
    df_undersampled = df_continua_excluido.group_by('foto_mes').apply(
        lambda group: group.sample(frac=0.2)
    ).flatten()


    df_undersampled_split = df_undersampled.filter(
    (pl.col('foto_mes').is_in(meses_totales))
)

    # Guardar el DataFrame en un archivo Parquet
    df_undersampled_split.write_parquet('/home/fililoco/buckets/b1/datasets/competencia_03_pre_sumas_lags_3_9_undersample_split.parquet')
