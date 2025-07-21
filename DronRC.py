# -*- coding: utf-8 -*-
import sys
assert sys.version_info >= (3, 7)
from packaging import version
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances_argmin
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.io
from scipy.io import loadmat
import zipfile
from pathlib import Path

"""Definicón de la clase DroneRC"""

class DronRC:
  def __init__(self, ruta_completa):

    self.ruta_completa = Path(ruta_completa)    
    self.nombre_archivo = self.ruta_completa.parts[-1]
    self.marca = self.nombre_archivo.split('_')[0]                       # Marca del dron
    self.modelo = self.nombre_archivo.split('_')[1]                      # Modelo del dron
    self.indice = self.nombre_archivo.split('_')[2].replace('.mat', '')  # Indice del archivo .mat del directorio correspondiente
    
    self.datosSinProcesar = self.load_data()            # Señal RF digitalizada: y[n]
    self.datosSinProcesar = self.datosSinProcesar.astype(np.float64) - np.mean(self.datosSinProcesar)

    self.factorEscala = None     # Factor de escala para convertir y[n] en voltios
    self.numMuestras = None      # Número de muestras en la señal capturada (5e6 muestras)
    self.duracion = None         # Tiempo que abarca la señal capturada y[n] (0.25 ms)
    self.Fs = None               # Frecuencia de muestreo del osciloscopio (20 GSa/s).
    self.datosReducidos = []   # Parte de los datos que incluye el transitorio de la señal para ahorrar almacenamiento
    self.caracteristicas = []  # Estructura que incluye las características extraídas
  

  def load_data(self):
    zip_path = r"D:\MPACT_DroneRC_RF_Dataset.zip"

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
      ruta_origen = rf"MPACT_DroneRC_RF_Dataset/{self.marca}_{self.modelo}/{self.marca}_{self.modelo}_{self.indice}.mat"
      ruta_destino = rf"C:\Users\joset\DroneRF_project"
      zip_ref.extract(ruta_origen, ruta_destino)

    ruta_mat_extraido = ruta_destino + '\\' rf"MPACT_DroneRC_RF_Dataset\{self.marca}_{self.modelo}\{self.marca}_{self.modelo}_{self.indice}.mat"
    mat = scipy.io.loadmat(ruta_mat_extraido)
    signal= mat['data'][0,0]['Data']
    return signal
  

 
