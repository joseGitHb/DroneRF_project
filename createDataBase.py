import os
import pandas as pd
from pathlib import Path
from DronRC import DronRC
import zipfile
import scipy.io
import warnings
import numpy as np

def createDataBase(zipPath, numSignals, opts, tableFormat=False):
    """
        Crea una base de datos a partir de los archivos .mat

        Argumentos:
               zipPath (str o Path): Ruta al .zip. 
               numSignals (int): Número de archivos a cargar por modelo. 
               opts (str): 'OnlyRawData', 'OnlyCroppedData', 'Features'
               tableFormat (bool): devuelve un dataFrame con las propiedaes

        Devuelve:
        list[list[DronRC]] o pd.DataFrame: La base de datos creada. 
    """

    zipPath = Path(zipPath)
    if not zipPath.exists():
        print(f"El archivo {zipPath} no existe")
    
    drone_objs = []
    with zipfile.ZipFile(zipPath, 'r') as z:
        all_files_inside_zip = z.namelist()
        # División de todas las rutas según '/' y selecciona la segunda parte. 
        # Ejemplo: MPACT_DroneRC_RF_Dataset/DJI_Inspire1Pro/DJI_Inspire1Pro_0001.mat' --> DJI_Inspire1Pro
        # set(...) elimina duplicados
        # sorted(...) los ordena alfabéticamente
        folders = sorted(set(
        f.split('/')[1] for f in all_files_inside_zip
        if f.startswith("MPACT_DroneRC_RF_Dataset/") and f.count('/') > 1
        ))
    
        
    for folder in folders:
        mat_files = [f for f in all_files_inside_zip
                     if f.startswith(f"MPACT_DroneRC_RF_Dataset/{folder}/") and f.endswith(".mat")]
        mat_files = sorted(mat_files)[:numSignals]
        for mat_file in mat_files:
            try:
                full_path = str(zipPath) + '\\' + mat_file.replace('/', '\\')
                dron = DronRC(full_path, opts)
                drone_objs.append(dron)
                #print(drone_objs)
            except Exception as e: 
                warnings.warn(f"Error procesando {mat_file}: {e}")
    
    if tableFormat:
        all_dicts = [vars(d) for d in drone_objs] # vars(d) devuelve un diccionario equivalente a un objeto DronRC
        df = pd.DataFrame(all_dicts) # Crea DataFrame de pandas a partir de los diccionarios
        return df
    else:
        return drone_objs
