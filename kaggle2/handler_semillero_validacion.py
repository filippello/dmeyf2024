import yaml
import subprocess
import os

def load_config(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config


# Definir la ruta del archivo config y el script a ejecutar
config_path = os.getenv("CONFIG_PATH")
config = load_config(config_path)
script1_path = "handler_prediction_single_validation.py"
script2_path = "handler_ganancia.py"  # Segundo script a ejecutar
results_path = "results.txt"  # Archivo donde se guardarán los resultados

def modify_config(seed_number):
    # Cargar el archivo YAML

    
    # Modificar el valor de la semilla en la posición especificada
    config['semillas'][0] = seed_number
    with open(config_path, "w") as file:
        yaml.dump(config, file)

def run_script(script_path):
    # Ejecutar el archivo .py usando subprocess y capturar la salida
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    print(f"Script {script_path} ejecutado con éxito.")
    return result.stdout  # Devolver la salida del script como resultado


if __name__ == "__main__":
    results = []

    # Leer el archivo de configuración inicial
    seed_original = config['semillas']
    seed_value = config['semillas'][0]
    semillero_count = config['semillero_count']
    # Loop de 1 a 30
    for i in range(1, semillero_count+1):
        # Incrementar los valores de semilla y study_number
        new_seed_value = seed_value + i
        modify_config(new_seed_value)

        # Ejecutar ambos scripts y almacenar los resultados
        result1 = run_script(script1_path)
        #result2 = run_script(script2_path)
        
        # Guardar el resultado de esta iteración en la lista
        results.append(f"Iteración {i} - Script1 Output: {result1.strip()}")
    modify_config(seed_original[0])
    # Escribir los resultados en un archivo
    with open(results_path, "w") as file:
        for result in results:
            file.write(result + "\n")

    print(f"Resultados guardados en {results_path}")
