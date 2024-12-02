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

    # Cargar el dataset
    dataset_input = '/home/fililoco/buckets/b1/datasets/competencia_03_pre_sumas.parquet'
    competencia_02 = pl.read_parquet(dataset_input)
    print(competencia_02.head())

    def division_segura(numerador, denominador):
        return pl.when(denominador == 0).then(0).otherwise(numerador / denominador)

    df = competencia_02

    # Definir lista de columnas de interés
    columnas = [
    "mrentabilidad","mrentabilidad_annual","mcomisiones","mactivos_margen","mpasivos_margen","cproductos","mcuenta_corriente",
    "ccaja_ahorro","mcaja_ahorro","mcaja_ahorro_adicional","mcaja_ahorro_dolares","mcuentas_saldo","ctarjeta_debito","ctarjeta_debito_transacciones",
    "mautoservicio","ctarjeta_visa_transacciones","mtarjeta_visa_consumo","ctarjeta_master_transacciones","mtarjeta_master_consumo",
    "cprestamos_personales","mprestamos_personales","cprestamos_prendarios","mprestamos_prendarios","cprestamos_hipotecarios","mprestamos_hipotecarios",
    "cplazo_fijo","mplazo_fijo_dolares","mplazo_fijo_pesos","cinversion1","minversion1_pesos","minversion1_dolares","cinversion2",
    "minversion2","mpayroll","mpayroll2","ccuenta_debitos_automaticos","mcuenta_debitos_automaticos","ctarjeta_visa_debitos_automaticos",
    "mttarjeta_master_debitos_automaticos","cpagodeservicios","mpagodeservicios","cpagomiscuentas","mpagomiscuentas","ccajeros_propios_descuentos","mcajeros_propios_descuentos",
    "ctarjeta_visa_descuentos","mtarjeta_visa_descuentos","ctarjeta_master_descuentos","mtarjeta_master_descuentos","ccomisiones_mantenimiento",
    "mcomisiones_mantenimiento","ccomisiones_otras","mcomisiones_otras","cforex","cforex_buy","mforex_buy","cforex_sell",
    "mforex_sell","ctransferencias_recibidas","mtransferencias_recibidas","ctransferencias_emitidas","mtransferencias_emitidas",
    "cextraccion_autoservicio","mextraccion_autoservicio","ccheques_depositados","mcheques_depositados","ccheques_emitidos","mcheques_emitidos",
    "ccheques_depositados_rechazados","mcheques_depositados_rechazados","ccheques_emitidos_rechazados","mcheques_emitidos_rechazados",
    "ccallcenter_transacciones","chomebanking_transacciones","ccajas_transacciones","ccajas_consultas","ccajas_depositos","ccajas_extracciones",
    "ccajas_otras","catm_trx","matm","catm_trx_other","matm_other","ctrx_quarter","Master_msaldototal",
    "Master_msaldopesos","Master_msaldodolares","Master_mconsumospesos","Master_mconsumosdolares","Master_mlimitecompra","Master_madelantopesos",
    "Master_madelantodolares","Master_mpagado","Master_mpagospesos","Master_mpagosdolares","Master_mconsumototal","Master_cconsumos",
    "Master_cadelantosefectivo","Visa_msaldototal","Visa_msaldopesos","Visa_msaldodolares","Visa_mconsumospesos","Visa_mconsumosdolares","Visa_mlimitecompra",
    "Visa_madelantopesos","Visa_madelantodolares","Visa_mpagospesos","Visa_mpagosdolares","Visa_mconsumototal","Visa_cconsumos","Visa_cadelantosefectivo",
] # Cambia estos nombres a tus columnas de interés

    # Ordenar por 'numero_de_cliente' y 'foto_mes' para asegurarse del orden cronológico por cliente
    df = df.sort(by=["numero_de_cliente", "foto_mes"])

    # Calcular las métricas por cliente para cada columna en la lista
    agg_columns = []
    for col in columnas:
        agg_columns.append(pl.col(col).max().alias(f"max_1m_{col}"))
        agg_columns.append(pl.col(col).min().alias(f"min_1m_{col}"))
        agg_columns.append(pl.col(col).mean().round(2).alias(f"avg_1m_{col}"))
        agg_columns.append(pl.col(col).max().alias(f"max_3m_{col}"))
        agg_columns.append(pl.col(col).min().alias(f"min_3m_{col}"))
        agg_columns.append(pl.col(col).mean().round(2).alias(f"avg_3m_{col}"))
        agg_columns.append(pl.col(col).max().alias(f"max_9m_{col}"))
        agg_columns.append(pl.col(col).min().alias(f"min_9m_{col}"))
        agg_columns.append(pl.col(col).mean().round(2).alias(f"avg_9m_{col}"))

    # Aplicar group_by() para calcular las métricas por cliente
    agg_df = df.group_by("numero_de_cliente").agg(agg_columns)

    # Unir los resultados agregados con el DataFrame original por 'numero_de_cliente'
    df = df.join(agg_df, on="numero_de_cliente", how="left")

    # Crear lags para 1 mes, 3 meses y 9 meses, pero por cada 'numero_de_cliente' para cada columna de la lista
    lag_columns = []
    for col in columnas:
        lag_columns.append(pl.col(col).shift(1).over("numero_de_cliente").round(2).alias(f"{col}_lag_1m"))
        lag_columns.append(pl.col(col).shift(3).over("numero_de_cliente").round(2).alias(f"{col}_lag_3m"))
        lag_columns.append(pl.col(col).shift(9).over("numero_de_cliente").round(2).alias(f"{col}_lag_9m"))

    df = df.with_columns(lag_columns)

    # Calcular delta entre el primer valor de la ventana y el actual para cada columna
    delta_columns = []
    for col in columnas:
        delta_columns.append((pl.col(col) - pl.col(f"{col}_lag_1m")).round(2).alias(f"delta_1m_{col}"))
        delta_columns.append((pl.col(col) - pl.col(f"{col}_lag_3m")).round(2).alias(f"delta_3m_{col}"))
        delta_columns.append((pl.col(col) - pl.col(f"{col}_lag_9m")).round(2).alias(f"delta_9m_{col}"))

    df = df.with_columns(delta_columns)

    # Mostrar el DataFrame resultante
    print(df)

    # Guardar el DataFrame en un archivo Parquet
    df.write_parquet('/home/fililoco/buckets/b1/datasets/competencia_03_pre_sumas_lags_3_9.parquet')
