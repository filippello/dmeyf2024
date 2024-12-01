dataset_input = '/home/fililoco/buckets/b1/datasets/competencia_03_pre.parquet'
import polars as pl

competencia_02 = pl.read_parquet(dataset_input)

competencia_02.head

import polars as pl
def division_segura(numerador, denominador):
    return np.where(denominador == 0, 0, numerador / denominador)

import numpy as np
df = competencia_02
# Suponiendo que tu DataFrame se llama 'df'
df = df.with_columns([
    (pl.col("mtarjeta_visa_consumo").fill_none(0) + pl.col("mtarjeta_master_consumo").fill_none(0)).alias("tc_consumo_total"),
    (pl.col("Master_mfinanciacion_limite").fill_none(0) + pl.col("Visa_mfinanciacion_limite").fill_none(0)).alias("tc_financiacionlimite_total"),
    (pl.col("Master_msaldopesos").fill_none(0) + pl.col("Visa_msaldopesos").fill_none(0)).alias("tc_saldopesos_total"),
    (pl.col("Master_msaldodolares").fill_none(0) + pl.col("Visa_msaldodolares").fill_none(0)).alias("tc_saldodolares_total"),
    (pl.col("Master_mconsumospesos").fill_none(0) + pl.col("Visa_mconsumospesos").fill_none(0)).alias("tc_consumopesos_total"),
    (pl.col("Master_mlimitecompra").fill_none(0) + pl.col("Visa_mlimitecompra").fill_none(0)).alias("tc_limitecompra_total"),
    (pl.col("Master_madelantopesos").fill_none(0) + pl.col("Visa_madelantopesos").fill_none(0)).alias("tc_adelantopesos_total"),
    (pl.col("Master_madelantodolares").fill_none(0) + pl.col("Visa_madelantodolares").fill_none(0)).alias("tc_adelantodolares_total"),
    (pl.col("tc_adelantopesos_total").fill_none(0) + pl.col("tc_adelantodolares_total").fill_none(0)).alias("tc_adelanto_total"),
    (pl.col("Master_mpagado").fill_none(0) + pl.col("Visa_mpagado").fill_none(0)).alias("tc_pagado_total"),
    (pl.col("Master_mpagospesos").fill_none(0) + pl.col("Visa_mpagospesos").fill_none(0)).alias("tc_pagadopesos_total"),
    (pl.col("Master_mpagosdolares").fill_none(0) + pl.col("Visa_mpagosdolares").fill_none(0)).alias("tc_pagadodolares_total"),
    (pl.col("Master_msaldototal").fill_none(0) + pl.col("Visa_msaldototal").fill_none(0)).alias("tc_saldototal_total"),
    (pl.col("Master_mconsumototal").fill_none(0) + pl.col("Visa_mconsumototal").fill_none(0)).alias("tc_consumototal_total"),
    (pl.col("Master_cconsumos").fill_none(0) + pl.col("Visa_cconsumos").fill_none(0)).alias("tc_cconsumos_total"),
    (pl.col("Master_delinquency").fill_none(0) + pl.col("Visa_delinquency").fill_none(0)).alias("tc_morosidad_total"),
    pl.max_horizontal(['Master_Fvencimiento', 'Visa_Fvencimiento']).alias('tc_fvencimiento_mayor'),
    pl.min_horizontal(['Master_Fvencimiento', 'Visa_Fvencimiento']).alias('tc_fvencimiento_menor'),
    pl.max_horizontal(['Master_fechaalta', 'Visa_fechaalta']).alias('tc_fechaalta_mayor'),
    pl.min_horizontal(['Master_fechaalta', 'Visa_fechaalta']).alias('tc_fechalta_menor'),
    pl.max_horizontal(['Master_Finiciomora', 'Visa_Finiciomora']).alias('tc_fechamora_mayor'),
    pl.min_horizontal(['Master_Finiciomora', 'Visa_Finiciomora']).alias('tc_fechamora_menor'),
    pl.max_horizontal(['Master_fultimo_cierre', 'Visa_fultimo_cierre']).alias('tc_fechacierre_mayor'),
    pl.min_horizontal(['Master_fultimo_cierre', 'Visa_fultimo_cierre']).alias('tc_fechacierre_menor'),
    (pl.col('mplazo_fijo_dolares').fill_null(0) + pl.col('mplazo_fijo_pesos').fill_null(0)).alias('m_plazofijo_total'),
    (pl.col('minversion1_dolares').fill_null(0) + pl.col('minversion1_pesos').fill_null(0)).alias('m_inversion1_total'),
    (pl.col('mpayroll').fill_null(0) + pl.col('mpayroll2').fill_null(0)).alias('m_payroll_total'),
    (pl.col('cpayroll_trx').fill_null(0) + pl.col('cpayroll2_trx').fill_null(0)).alias('c_payroll_total')
        (pl.col('mplazo_fijo_dolares').fill_null(0) + pl.col('mplazo_fijo_pesos').fill_null(0)).alias('m_plazofijo_total'),
    (pl.col('minversion1_dolares').fill_null(0) + pl.col('minversion1_pesos').fill_null(0)).alias('m_inversion1_total'),
    (pl.col('mpayroll').fill_null(0) + pl.col('mpayroll2').fill_null(0)).alias('m_payroll_total'),
    (pl.col('cpayroll_trx').fill_null(0) + pl.col('cpayroll2_trx').fill_null(0)).alias('c_payroll_total'),
    division_segura(pl.col('m_plazofijo_total'), pl.col('cplazo_fijo')).alias('m_promedio_plazofijo_total'),
    division_segura(pl.col('m_inversion1_total'), pl.col('cinversion1')).alias('m_promedio_inversion_total'),
    division_segura(pl.col('mcaja_ahorro'), pl.col('ccaja_ahorro')).alias('m_promedio_caja_ahorro'),
    division_segura(pl.col('mtarjeta_visa_consumo'), pl.col('ctarjeta_visa_transacciones')).alias('m_promedio_tarjeta_visa_consumo_por_transaccion'),
    division_segura(pl.col('mtarjeta_master_consumo'), pl.col('ctarjeta_master_transacciones')).alias('m_promedio_tarjeta_master_consumo_por_transaccion'),
    division_segura(pl.col('mprestamos_prendarios'), pl.col('cprestamos_prendarios')).alias('m_promedio_prestamos_prendarios'),
    division_segura(pl.col('mprestamos_hipotecarios'), pl.col('cprestamos_hipotecarios')).alias('m_promedio_prestamos_hipotecarios'),
    division_segura(pl.col('minversion2'), pl.col('cinversion2')).alias('m_promedio_inversion2'),
    division_segura(pl.col('mpagodeservicios'), pl.col('cpagodeservicios')).alias('m_promedio_pagodeservicios'),
    division_segura(pl.col('mpagomiscuentas'), pl.col('cpagomiscuentas')).alias('m_promedio_pagomiscuentas'),
    division_segura(pl.col('mcajeros_propios_descuentos'), pl.col('ccajeros_propios_descuentos')).alias('m_promedio_cajeros_propios_descuentos'),
    division_segura(pl.col('mtarjeta_visa_descuentos'), pl.col('ctarjeta_visa_descuentos')).alias('m_promedio_tarjeta_visa_descuentos'),
    division_segura(pl.col('mtarjeta_master_descuentos'), pl.col('ctarjeta_master_descuentos')).alias('m_promedio_tarjeta_master_descuentos'),
    division_segura(pl.col('mcomisiones_mantenimiento'), pl.col('ccomisiones_mantenimiento')).alias('m_promedio_comisiones_mantenimiento'),
    division_segura(pl.col('mcomisiones_otras'), pl.col('ccomisiones_otras')).alias('m_promedio_comisiones_otras'),
    division_segura(pl.col('mforex_buy'), pl.col('cforex_buy')).alias('m_promedio_forex_buy'),
    division_segura(pl.col('mtransferencias_recibidas'), pl.col('ctransferencias_recibidas')).alias('m_promedio_transferencias_recibidas'),
    division_segura(pl.col('mtransferencias_emitidas'), pl.col('ctransferencias_emitidas')).alias('m_promedio_transferencias_emitidas'),
    division_segura(pl.col('mextraccion_autoservicio'), pl.col('cextraccion_autoservicio')).alias('m_promedio_extraccion_autoservicio'),
    division_segura(pl.col('mcheques_depositados'), pl.col('ccheques_depositados')).alias('m_promedio_cheques_depositados'),
    division_segura(pl.col('mcheques_emitidos'), pl.col('ccheques_emitidos')).alias('m_promedio_cheques_emitidos'),
    division_segura(pl.col('mcheques_depositados_rechazados'), pl.col('ccheques_depositados_rechazados')).alias('m_promedio_cheques_depositados_rechazados'),
    division_segura(pl.col('mcheques_emitidos_rechazados'), pl.col('ccheques_emitidos_rechazados')).alias('m_promedio_cheques_emitidos_rechazados'),
    division_segura(pl.col('matm'), pl.col('catm_trx')).alias('m_promedio_atm'),
    division_segura(pl.col('matm_other'), pl.col('catm_trx_other')).alias('m_promedio_atm_other'),
    division_segura(pl.col('Master_msaldototal'), pl.col('Master_mfinanciacion_limite')).alias('proporcion_financiacion_master_cubierto'),
    division_segura(pl.col('Master_msaldototal'), pl.col('Master_mlimitecompra')).alias('proporcion_limite_master_cubierto'),
    division_segura(pl.col('Visa_msaldototal'), pl.col('Visa_mfinanciacion_limite')).alias('proporcion_financiacion_visa_cubierto'),
    division_segura(pl.col('Visa_msaldototal'), pl.col('Visa_mlimitecompra')).alias('proporcion_limite_visa_cubierto'),
    division_segura(pl.col('tc_saldototal_total'), pl.col('tc_financiacionlimite_total')).alias('proporcion_financiacion_total_cubierto'),
    division_segura(pl.col('tc_saldototal_total'), pl.col('tc_limitecompra_total')).alias('proporcion_limite_total_cubierto'),
    division_segura(pl.col('tc_saldopesos_total'), pl.col('tc_saldototal_total')).alias('tc_proporcion_saldo_pesos'),
    division_segura(pl.col('tc_saldodolares_total'), pl.col('tc_saldototal_total')).alias('tc_proporcion_saldo_dolares'),
    division_segura(pl.col('tc_consumopesos_total'), pl.col('tc_consumototal_total')).alias('tc_proporcion_consumo_pesos')
])

df.to_parquet('/home/fililoco/buckets/b1/datasets/competencia_03_pre_sumas.parquet', index=False)