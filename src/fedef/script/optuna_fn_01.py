import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice, plot_contour

#a. Voy a realizar un estudio de Optuna para encontrar los mejores parámetros.
#i. Creo la base de datos donde guardar los resultados.
def optuna_fn():
	storage_name = "sqlite:///" + db_path + "optimization_lgbm.db"
	study_name = f"{study_type}_{study_number}_{study_protocol}_data-{study_data}_optuna-{study_optuna}_timeframe{study_timeframe}_extra-{study_aditional}"
	#study_name = "exp_300_lgbm_datos_crudos_100_num_boost_round"

	#ii. Creo el estudio.
	study = optuna.create_study(
    	direction="maximize",
    	study_name=study_name,
    	storage=storage_name,
    	load_if_exists=True,
	)	

	#iii. Corro el estudio.
	study.optimize(objective, n_trials)
