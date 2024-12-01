import polars as pl
import numpy as np


dataset_input = '/home/fililoco/buckets/b1/datasets/competencia_03_pre.parquet'
# Lee el archivo CSV usando Polars
competencia_02 = pl.read_csv(dataset_input)

# Verifica los datos cargados
print(competencia_02.head())  # Muestra las primeras filas

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

# Ordenar los datos por cliente y mes
competencia_02 = competencia_02.sort(["numero_de_cliente", "foto_mes"])



# Ventanas en meses
ventanas = [3, 6]

# Lista de columnas dinámicas
columnas_dinamicas = campos_iniciales

# Nombre de las columnas que identifican cliente y tiempo
cliente_col = "numero_de_cliente"
fecha_col = "foto_mes"

# Función para calcular la pendiente en una ventana
def calculate_slope(series):
    x = np.arange(len(series))
    slope, _ = np.polyfit(x, series, 1)
    return slope

# Lista para almacenar todas las transformaciones
transformaciones = []


for campo in columnas_dinamicas:
    for ventana in ventanas:
        # Agregamos estadísticas básicas y avanzadas
        transformaciones.extend([
            pl.col(campo).rolling_mean(window_size=ventana).over(cliente_col).round(3).alias(f"{campo}_avg_{ventana}m"),
            pl.col(campo).rolling_sum(window_size=ventana).over(cliente_col).round(3).alias(f"{campo}_sum_{ventana}m"),
            pl.col(campo).rolling_max(window_size=ventana).over(cliente_col).round(3).alias(f"{campo}_max_{ventana}m"),
            pl.col(campo).rolling_min(window_size=ventana).over(cliente_col).round(3).alias(f"{campo}_min_{ventana}m"),
            pl.col(campo).rolling_std(window_size=ventana).over(cliente_col).round(3).alias(f"{campo}_volatility_{ventana}m"),
            (pl.col(campo).rolling_max(window_size=ventana).over(cliente_col) /
             pl.col(campo).rolling_min(window_size=ventana).over(cliente_col)).round(3).alias(f"{campo}_range_ratio_{ventana}m"),
            # Deltas y tasas de cambio
            (pl.col(campo) - pl.col(campo).shift(ventana).over(cliente_col)).round(3).alias(f"{campo}_delta_{ventana}m"),
            ((pl.col(campo) - pl.col(campo).shift(ventana).over(cliente_col)) /
             pl.col(campo).shift(ventana).over(cliente_col)).round(3).alias(f"{campo}_rate_change_{ventana}m")
        ])

        # Calcular pendiente como una operación personalizada
        transformaciones.append(
            pl.col(campo)
            .rolling_apply(window_size=ventana, function=calculate_slope)
            .over(cliente_col)
            .round(3)
            .alias(f"{campo}_slope_{ventana}m")
        )

    # Agregar shift del mes anterior
    transformaciones.append(
        pl.col(campo).shift(1).over(cliente_col).round(3).alias(f"{campo}_prev_1m")
    )



# Aplicamos las transformaciones al DataFrame
competencia_02 = competencia_02.with_columns(transformaciones)

# Mostrar el resultado
print(competencia_02)


# Agregar las nuevas columnas al dataframe original
# Suponiendo que `columnas_dinamicas` es una lista de expresiones Polars para agregar nuevas columnas
competencia_02_sumas_drifting = competencia_02.with_columns(*columnas_dinamicas)

# Verificar los resultados
print(competencia_02_sumas_drifting.head())
mes_train_ult_6_meses = [201909, 202003, 202009,202103,202109]


# Filtrar las filas donde 'foto_mes' está en la lista `mes_train_ult_6_meses`
competencia_02_sumas_drifting = competencia_02_sumas_drifting.filter(
    pl.col('foto_mes').is_in(mes_train_ult_6_meses)
)

# Ruta de salida
dataset_output = '/home/fililoco/buckets/b1/datasets/competencia_03_pre_lags.parquet'

# Exportar a CSV
competencia_02_sumas_drifting.write_csv(dataset_output)
