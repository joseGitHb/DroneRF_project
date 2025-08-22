import os
import pandas as pd
from pathlib import Path
from DronRC import DronRC
import zipfile
import scipy.io
import warnings
import numpy as np


def createDataBase(rootPath, numSignals, opts, tableFormat):
    """
        Crea una base de datos a partir de los archivos .mat

        Argumentos:
               rootPath(str o Path): Ruta a la subcarpeta raíz. 
               numSignals (int): Número de ficheros (señales) a cargar por modelo. 
               opts (str): 'OnlyRawData', 'OnlyCroppedData', 'IncludeFeatures'
               tableFormat (bool): devuelve un dataFrame con las propiedaes
        Devuelve:
            list[list[DronRC]] o pd.DataFrame: La base de datos creada. 
    """
    rootPath = Path(rootPath)
    if not rootPath.exists():
        print(f"La carpeta {rootPath} no existe")

    drone_objs = []
    folders = sorted([f for f in rootPath.iterdir() if f.is_dir()])

    for folder in folders:
        files = [file for file in folder.iterdir()]
        for mat_file in files[:numSignals]:
            try:                   
                dron = DronRC(str(mat_file), opts)
                drone_objs.append(dron)        
            except Exception as e:            
                warnings.warn(f"Error procesando {mat_file}: {e}") 

    if tableFormat:
        all_dicts = [vars(d) for d in drone_objs]
        df = pd.DataFrame(all_dicts)
        return df
    else:
        return drone_objs

