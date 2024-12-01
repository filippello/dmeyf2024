dataset_input = '/home/fililoco/buckets/b1/datasets/competencia_03_pre.parquet'

import polars as pl

competencia_02 = pl.read_parquet(dataset_input)

campos_iniciales = [
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
]

import polars as pl


# Ordenar los datos por cliente y mes
competencia_02 = competencia_02.sort(["numero_de_cliente", "foto_mes"])

# Crear nuevas columnas dinámicamente
columnas_dinamicas = []

for campo in campos_iniciales:
    # Ventana de 12 meses
    columnas_dinamicas.extend([
        pl.col(campo).rolling_mean(window_size=12).over("numero_de_cliente").alias(f"{campo}_avg_12m"),
        pl.col(campo).rolling_max(window_size=12).over("numero_de_cliente").alias(f"{campo}_max_12m"),
        pl.col(campo).rolling_min(window_size=12).over("numero_de_cliente").alias(f"{campo}_min_12m"),
        pl.col(campo).rolling_sum(window_size=12).over("numero_de_cliente").alias(f"{campo}_count_12m"),
        (pl.col(campo) * pl.col("foto_mes")).rolling_mean(window_size=12).over("numero_de_cliente").alias(f"{campo}_slope_12m"),
    ])
    # Ventana de 6 meses
    columnas_dinamicas.extend([
        pl.col(campo).rolling_mean(window_size=6).over("numero_de_cliente").alias(f"{campo}_avg_6m"),
        pl.col(campo).rolling_max(window_size=6).over("numero_de_cliente").alias(f"{campo}_max_6m"),
        pl.col(campo).rolling_min(window_size=6).over("numero_de_cliente").alias(f"{campo}_min_6m"),
        pl.col(campo).rolling_sum(window_size=6).over("numero_de_cliente").alias(f"{campo}_count_6m"),
        (pl.col(campo) * pl.col("foto_mes")).rolling_mean(window_size=6).over("numero_de_cliente").alias(f"{campo}_slope_6m"),
    ])
    # Ventana de 3 meses
    columnas_dinamicas.extend([
        pl.col(campo).rolling_mean(window_size=3).over("numero_de_cliente").alias(f"{campo}_avg_3m"),
        pl.col(campo).rolling_max(window_size=3).over("numero_de_cliente").alias(f"{campo}_max_3m"),
        pl.col(campo).rolling_min(window_size=3).over("numero_de_cliente").alias(f"{campo}_min_3m"),
        pl.col(campo).rolling_sum(window_size=3).over("numero_de_cliente").alias(f"{campo}_count_3m"),
        (pl.col(campo) * pl.col("foto_mes")).rolling_mean(window_size=3).over("numero_de_cliente").alias(f"{campo}_slope_3m"),
    ])
    # Mes anterior (LAG)
    columnas_dinamicas.append(
        pl.col(campo).shift(1).over("numero_de_cliente").alias(f"{campo}_prev_1m")
    )

# Agregar las nuevas columnas al dataframe original
competencia_02_sumas_drifting = competencia_02.with_columns(columnas_dinamicas)

# Verificar los resultados
print(competencia_02_sumas_drifting.head())


competencia_02_sumas_drifting.to_parquet('/home/fililoco/buckets/b1/datasets/competencia_03_test1.parquet', index=False)