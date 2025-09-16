
import pandas as pd
from pathlib import Path
from DronRC import DronRC
import warnings

def createDataBase(rootPath, numSignals, opts, tableFormat, start_file):
    """
        Crea una base de datos a partir de los archivos MATLAB. 

        Argumentos:
               rootPath (str): Ruta de la carpeta que contiene las subcarpetas con los archivos .mat.
               numSignals (int): Número de ficheros (señales) a cargar por modelo. 
               opts (str): 'OnlyRawData', 'OnlyCroppedData', 'IncludeFeatures'
               tableFormat (bool): devuelve un dataFrame con las propiedaes si es True.
               start_file (int): índice del primer archivo a cargar en cada carpeta.
        Devuelve:
            list[list[DronRC]] o pd.DataFrame: La base de datos creada. 
    """
    rootPath = Path(rootPath) # Convertir a objeto Path     
    if not rootPath.exists(): # Comprobar que la ruta existe
        print(f"La carpeta {rootPath} no existe")

    drone_objs = []  # Lista para almacenar los objetos DronRC   
    folders = sorted([f for f in rootPath.iterdir() if f.is_dir()])  # Lista de las carpetas (tipos de drones) 

    for folder in folders:                          # Recorrer cada carpeta (tipo de dron)
        files = [file for file in folder.iterdir()] # Lista de archivos .mat en la carpeta actual
        for mat_file in files[start_file:start_file+numSignals]: # Recorrer desde start_file hasta start_file+numSignals
            try:                   
                dron = DronRC(str(mat_file), opts) # Crear el objeto DronRC para el archivo actual
                drone_objs.append(dron)            # Añadir el objeto a la lista
            except Exception as e:            
                warnings.warn(f"Error procesando {mat_file}: {e}") 

    if tableFormat:        
        all_dicts = [vars(d) for d in drone_objs]  # Cada elemento de df es un diccionario con las propiedades de un objeto DronRC
        df = pd.DataFrame(all_dicts) # Crear el DataFrame a partir de la lista de diccionarios
        return df
    else:
        return drone_objs

