from pathlib import Path
import zipfile
import warnings

zip_path = Path(r"D:\MPACT_DroneRC_RF_Dataset.zip") # Ruta origen de la base de datos comprimida
target_path = Path(r"E:\DataBase") # Ruta destino donde se extraerán los archivos .mat

with zipfile.ZipFile(zip_path, 'r') as z: # Abrir el archivo zip en modo lectura
        all_files_inside_zip = z.namelist()  # Lista de todos los elementos dentro del zip  
        # folders: lista de carpetas (tipos de drones) dentro del zip
        folders = sorted(set(
        f.split('/')[1] for f in all_files_inside_zip
        if f.startswith("MPACT_DroneRC_RF_Dataset/") and f.count('/') > 1
        ))

        for folder in folders:
                # mat_files: lista de archivos .mat dentro de la carpeta actual
                mat_files = [f for f in all_files_inside_zip
                            if f.startswith(f"MPACT_DroneRC_RF_Dataset/{folder}/") and f.endswith(".mat")]                
                mat_files = sorted(mat_files)[:700] # Limitar a 700 archivos por carpeta para no saturar la memoria externa
                for mat_file in mat_files:
                    try:
                        z.extract(mat_file, target_path) # Extraer el archivo .mat a la ruta destino
                        print("Procesando...")
                    except Exception as e: 
                        warnings.warn(f"Error procesando {mat_file}: {e}")        
