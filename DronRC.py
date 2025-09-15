# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from pathlib import Path
from scipy.signal import hilbert
from scipy.stats import skew, kurtosis, entropy

"""Definición de la clase DroneRC"""
class DronRC:
  """
    La clase DronRC se usa para crear un objeto que represente a cada archivo MATLAB de la base de datos
    MPACT_DroneRC_RF_Dataset. Cada objeto DronRC contiene los datos de la señal de RF, así como
    las características estadísticas calculadas a partir de la señal recortada.
  """
  def __init__(self, full_path, *args):
    self.full_path = Path(full_path)    # Ruta completa del archivo .mat
    self.file_name = self.full_path.parts[-1] # Nombre del archivo .mat
    if np.size(self.file_name.split('_')) == 3:  
      # Formato make_model_index.mat  (15 drones distintos)     
      self.make = self.file_name.split('_')[0]       # Marca                 
      self.model = self.file_name.split('_')[1]      # Modelo            
      self.index = self.file_name.split('_')[2].replace('.mat', '')  # Índice del archivo dentro del directorio asociado al dron
    else:                                         
      # Formato make_model_number_index.mat (Un par de drones repetidos)
      self.make = self.file_name.split('_')[0]       # Marca                      
      self.model = f"{self.file_name.split('_')[1]}_{self.file_name.split('_')[2]}"  # Modelo
      self.index = self.file_name.split('_')[3].replace('.mat', '')  # Índice del archivo
    
    struct_mat = self.load_struct_mat()       # Estructura del archivo .mat
    self.scaleFactor = struct_mat['YInc']     # Factor de escala: convierte la señal digitalizada a voltios
    self.numSamples = struct_mat['NumPoints'] # Número de muestras: 5 millones
    self.Fs = int(20e9)                       # Frecuencia de muestreo: 20 GHz         
    self.duration = self.numSamples/self.Fs   # Duración de la señal: 0.25 segundos   
    self.rawData = struct_mat['Data']         # Señal RF digitalizada  
    self.rawData = self.rawData.astype(np.float64) - np.mean(self.rawData) # Restamos la media
    self.rawData = np.array(self.rawData).ravel()  # Todas las muestras (5 millones) en formato array 1D
    # Asumimos que el transitorio se encuentra entre las muestras 2e6 y 3e6
    crop_start = int(2e6)
    crop_end = int(3e6)
    self.croppedData = self.rawData[crop_start:crop_end]   # Porción de la señal que contiene el transitorio
    self.croppedData = np.array(self.croppedData).ravel()  # Array 1D

    if not args or args[0]=='OnlyRawData':
      self.croppedData = [] # Se almacena sólo la señal original. No se calcula nada más.
    elif args[0] == 'OnlyCroppedData':
      self.rawData = []     # Se almacena sólo la señal recortada. No se calcula nada más.
    elif args[0] == 'IncludeFeatures':
      self.rawData = []     # Se almacena la señal recortada y se calculan las características estadísticas.
      
      self.transient_start = self.get_transient_start(self.croppedData) # Cálculo del inicio del transitorio
      x = self.croppedData[self.transient_start::10]  # Submuestreamos la señal cada 10 muestras. Ahorra memoria.
      abs_x = np.abs(x)  # Valor absoluto de la señal
      # Cálculo de las features (características estadísticas)
      self.Mean = float(np.mean(x))    # Media          
      abs_mean = float(np.mean(abs_x)) # Media del valor absoluto. No forma parte de las entradas del modelo
      self.StandardDeviation = float(np.std(x)) # Desviación estándar
      self.Skewness = float(skew(x))   # Asimetría
      self.Entropy = float(entropy(x))  # Entropía
      self.RootMeanSquare = float(np.sqrt(np.mean(x**2)))  # Valor eficaz (RMS)
      self.Root = float(np.mean(np.sqrt(abs_x))**2)  # Raíz
      self.Kurtosis = float(kurtosis(x)) # Descriptor de forma / Cola
      self.Variance = float(np.var(x))   # Varianza
      self.Peak = float(np.max(x))  # Valor pico
      self.Peak2Peak = float(np.max(x) - np.min(x)) # Valor pico a pico
      self.ShapeFactor = float(self.RootMeanSquare / abs_mean)  # Factor de forma
      self.CrestFactor = float(self.Peak / self.RootMeanSquare) # Factor de cresta
      self.ImpulseFactor = float(self.Peak / abs_mean)  # Factor de impulso
      self.ClearanceFactor = float(self.Peak / self.Root) # Factor de despeje
    else:
      raise ValueError('Invalid Option!')

  def load_struct_mat(self):   
    """Carga el archivo .mat y devuelve la estructura de datos"""
    mat = scipy.io.loadmat(self.full_path)
    struct_mat = mat['data'][0,0]
    return struct_mat
  
  def get_transient_start(self, cropData, plot=False):
    """
        Calcula el índice de inicio del transitorio en la señal recortada usando una técnica
        de detección de fase.
        Argumentos:
            cropData (array): Señal recortada.
            plot (bool): Si es True, genera una gráfica de la señal con el inicio del transitorio marcado.
        Devuelve:
            transient_start (int): Índice de inicio del transitorio.
    """
    cropData = np.array(cropData) # Aseguramos que cropData es un array numpy
    signal = hilbert(cropData)    # Transformada de Hilbert. Genera la señal analítica con parte real e imaginaria
    phi = np.angle(signal)        # Fase instantánea de la señal analítica
    av = np.unwrap(phi)           # Fase "desenvuelta" (crece linealmente sin saltos bruscos de 2π)
    N = np.size(cropData)         # Tamaño de la señal recortada (1 millón de muestras)  
    window_size = 100             # Tamaño de las ventanas no superpuestas en la que se calcula la varianza. 
    tv = []                       # Vector temporal (tv) que almacena la varianza de cada ventana
    num_windows = N//window_size  # Número total de ventanas (tamaño de tv[])

    for i in range(num_windows):
        g = (i+1)*window_size     # Final de la i-ésima ventana
        d = g - window_size       # Inicio de la i-ésima ventana
        window_i = av[d:g]        # Extraemos la i-ésima ventana de la fase desenvuelta
        tv.append(np.var(window_i)) # Calculamos y almacenamos la varianza de la i-ésima ventana
    
    ft = np.abs(np.diff(tv))   # Trayectoria Fractal (ft): diferencia entre varianzas de fase consecutivas
    ft = np.array(ft)         # Aseguramos que ft es un array numpy
    ft_size = np.size(ft)     # Tamaño del array ft (número de ventanas - 1)
    transient_window = None   # Índice de la ventana donde comienza el transitorio
    for i in range(ft_size-4):
        # Buscamos la primera ventana donde cinco valores consecutivos de ft son menores o iguales al umbral 5
        if ft[i]<=5 and ft[i+1]<=5 and ft[i+2]<=5 and ft[i+3]<=5 and ft[i+4]<=5:
            transient_window = i 
            break     
          
    transient_start = transient_window*window_size # Convertimos el índice de ventana al índice de muestra en croppedData

    if plot:
      time_ms = np.arange(np.size(self.croppedData)) / self.Fs * int(1e3) # Tiempo en ms
      plt.figure() 
      plt.plot(time_ms, np.array(self.croppedData*self.scaleFactor).ravel()) # Señal en voltios
      plt.xlabel('Time (ms)')
      plt.ylabel('Amplitude (V)')
      title = f"{self.make}_{self.model}_{self.index}" # Título de la gráfica
      plt.title(title)
      plt.grid(True)      
      plt.tight_layout()
      transient_start_ms = transient_start / self.Fs * 1e3       # Tiempo en ms del inicio del transitorio   
      plt.axvline(transient_start_ms, color='r', linewidth=1.0, label='Transient Start') # Línea vertical
      plt.savefig("transient_start.png")  
        
    return transient_start
  


