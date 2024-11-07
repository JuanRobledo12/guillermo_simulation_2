# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 08:22:24 2024

@author: guill
"""

import os
import json
import pandas as pd
import re
import time
import sys

# Handle command-line arguments for parallelization
if len(sys.argv) != 3:
    print("Usage: python script.py <start_index> <end_index>")
    sys.exit(1)

try:
    start_index = int(sys.argv[1])
    end_index = int(sys.argv[2])
except ValueError:
    print("start_index and end_index must be integers")
    sys.exit(1)

start_time = time.time()

# Definir los escenarios de demanda y oferta
reemplazos_demanda = [
    "ANC02_CLIMIN_ECOACE", "ANC02_CLIMIN_ECODIN", "ANC02_CLIMIN_ECOHIS",
    "ANC02_CLIMRCP45_ECOACE", "ANC02_CLIMRCP45_ECODIN", "ANC02_CLIMRCP45_ECOHIS",
    "ANC02_CLIMRCP85_ECOACE", "ANC02_CLIMRCP85_ECODIN", "ANC02_CLIMRCP85_ECOHIS",
    "ANC22_CLIMIN_ECOACE", "ANC22_CLIMIN_ECODIN", "ANC22_CLIMIN_ECOHIS",
    "ANC22_CLIMRCP45_ECOACE", "ANC22_CLIMRCP45_ECODIN", "ANC22_CLIMRCP45_ECOHIS",
    "ANC22_CLIMRCP85_ECOACE", "ANC22_CLIMRCP85_ECODIN", "ANC22_CLIMRCP85_ECOHIS",
    "ANCTEND_CLIMIN_ECOACE", "ANCTEND_CLIMIN_ECODIN", "ANCTEND_CLIMIN_ECOHIS",
    "ANCTEND_CLIMRCP45_ECOACE", "ANCTEND_CLIMRCP45_ECODIN", "ANCTEND_CLIMRCP45_ECOHIS",
    "ANCTEND_CLIMRCP85_ECOACE", "ANCTEND_CLIMRCP85_ECODIN", "ANCTEND_CLIMRCP85_ECOHIS"
]

reemplazos_oferta = [
    "1A, 0.90", "1A, 0.95", "1A, 0.97", "2A, 0.90", "2A, 0.95", "2A, 0.97",
    "3A, 0.90", "3A, 0.95", "3A, 0.97", "3B, 0.90", "3B, 0.95", "3B, 0.97",
    "4A, 0.90", "4A, 0.95", "4A, 0.97", "4B, 0.90", "4B, 0.95", "4B, 0.97",
    "5A, 0.90", "5A, 0.95", "5A, 0.97", "5B, 0.90", "5B, 0.95", "5B, 0.97",
    "6A, 0.90", "6A, 0.95", "6A, 0.97", "6B, 0.90", "6B, 0.95", "6B, 0.97",
    "6C, 0.90", "6C, 0.95", "6C, 0.97", "7A, 0.90", "7A, 0.95", "7A, 0.97"
]

# Definir la carpeta de trabajo donde están los archivos CSV
csv_folder_path = "model_results_server"
csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]

# Process only a subset based on the provided indices
csv_files_subset = csv_files[start_index:end_index]

# Definir la carpeta de salida
output_folder_base = "json_RDM"

# Check if the folder exists, and if not, create it
if not os.path.exists(output_folder_base):
    os.makedirs(output_folder_base)
    print(f"Folder '{output_folder_base}' created.")

# Inicializar contador para las carpetas de salida
output_folder_counter = start_index + 1
print('Generando archivos JSON...')
# Procesar cada archivo CSV en la parte seleccionada
for csv_file in csv_files_subset:
    csv_path = os.path.join(csv_folder_path, csv_file)
    csv_data = pd.read_csv(csv_path)
    
    # Limpiar los nombres de las columnas, eliminando paréntesis inicial, comillas extra y la parte final ', 0)'
    csv_data.columns = [re.sub(r"^\('|',? 0\)$|'$", "", col) for col in csv_data.columns]

    # Filtrar columnas de activación
    activacion_columns = [col for col in csv_data.columns if re.match(r'^par_activacion_p[0-9]+$', col, re.IGNORECASE)]
    
    # Crear carpeta de salida para este archivo CSV
    output_folder_name = f"po{output_folder_counter}"
    output_folder_path = os.path.join(output_folder_base, output_folder_name)
    os.makedirs(output_folder_path, exist_ok=True)
    output_folder_counter += 1

    # Cargar el archivo JSON base
    with open("OptModel_RDMv1.7.json", "r") as f:
        json_data = json.load(f)
        
    
    # Generar archivos JSON para cada combinación de oferta y demanda
    for demanda in reemplazos_demanda:
        for oferta in reemplazos_oferta:
            modified_json = json.loads(json.dumps(json_data))  # Clonar el JSON original
            
            # Actualizar valores de activación en el JSON
            for column in activacion_columns:
                project_index = re.search(r'P(\d+)', column).group(1)  # Esto captura solo el número después de la "P"
                param_name = f"activacion_P{project_index}"
                
                if param_name in modified_json['parameters']:
                    windows = modified_json['parameters'][param_name]['windows']
                    adjusted_windows = windows[:-1] + [windows[-1] - 1]
                    activacion_values = csv_data[column].iloc[adjusted_windows].tolist()
                    modified_json['parameters'][param_name]['values'] = activacion_values

            # Reemplazar los textos de escenarios de demanda y oferta
            modified_json_str = json.dumps(modified_json).replace("ANCTEND_CLIMIN_ECOACE", demanda).replace("1A, 0.95", oferta)
            modified_json = json.loads(modified_json_str)

            # Crear nombre de archivo de salida
            oferta_modificada = oferta.replace(", ", "_")
            output_filename = f"{output_folder_name}_{demanda}_{oferta_modificada}.json"
            output_path = os.path.join(output_folder_path, output_filename)

            # Guardar el archivo JSON modificado
            with open(output_path, "w") as f:
                json.dump(modified_json, f, indent=4)
            
    print(f"Directorio generado con archivos JSON")      
print("Proceso terminado!")
finish_time = time.time()
print(f"Total execution time: {finish_time - start_time} secs")
