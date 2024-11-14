import yaml
import subprocess

# Definir la ruta del archivo config y el script a ejecutar
config_path = "config.yaml"
script1_path = "handler_optuna.py"
script2_path = "handler_ganancia.py"  # Segundo script a ejecutar
results_path = "results.txt"  # Archivo donde se guardarán los resultados

def modify_config(seed_number, new_study_number):
    # Cargar el archivo YAML

    
    # Modificar el valor de la semilla en la posición especificada
    config['semillas'][0] = seed_number
    
    # Modificar el study_number
    config['study_number'] = str(new_study_number)

    # Guardar los cambios en el archivo YAML
    with open(config_path, "w") as file:
        yaml.dump(config, file)

    print(f"Archivo config.yaml actualizado: semilla={config['semillas'][0]}, study_number={new_study_number}")

def run_script(script_path):
    # Ejecutar el archivo .py usando subprocess y capturar la salida
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    print(f"Script {script_path} ejecutado con éxito.")
    return result.stdout  # Devolver la salida del script como resultado


if __name__ == "__main__":
    results = []

    # Leer el archivo de configuración inicial
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    seed_value = config['semillas'][0]
    study_number = int(config['study_number'])
    # Loop de 1 a 30
    for i in range(1, 31):
        # Incrementar los valores de semilla y study_number
        new_seed_value = seed_value + i
        new_study_number = study_number+ i

        # Modificar config.yaml
        modify_config(new_seed_value, new_study_number)

        # Ejecutar ambos scripts y almacenar los resultados
        result1 = run_script(script1_path)
        result2 = run_script(script2_path)
        
        # Guardar el resultado de esta iteración en la lista
        results.append(f"Iteración {i} - Script1 Output: {result1.strip()}, Script2 Output: {result2.strip()}")

    # Escribir los resultados en un archivo
    with open(results_path, "w") as file:
        for result in results:
            file.write(result + "\n")

    print(f"Resultados guardados en {results_path}")
