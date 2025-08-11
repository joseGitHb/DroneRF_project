# -*- coding: utf-8 -*-
import sys
assert sys.version_info >= (3, 7)
from packaging import version
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances_argmin
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import ruptures as rpt
import scipy.io
from scipy.io import loadmat
from scipy.signal import spectrogram
import zipfile
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert

"""Definicón de la clase DroneRC"""

class DronRC:
  def __init__(self, full_path, *args):

    self.full_path = Path(full_path)    
    self.file_name = self.full_path.parts[-1]
    if np.size(self.file_name.split('_')) == 3:                      # Para trabajar con 13 modelos
      self.make = self.file_name.split('_')[0]                       # Marca del dron
      self.model = self.file_name.split('_')[1]                      # Modelo del dron
      self.index = self.file_name.split('_')[2].replace('.mat', '')  # Indice del archivo .mat del directorio correspondiente
    else:
      self.make = self.file_name.split('_')[0]                       # Marca del dron
      self.model = f"{self.file_name.split('_')[1]}_{self.file_name.split('_')[2]}"     # Modelo del dron
      self.index = self.file_name.split('_')[3].replace('.mat', '')  
    
    self.rawData = self.load_struct_mat()['Data']            # Señal RF digitalizada: y[n]
    self.rawData = self.rawData.astype(np.float64) - np.mean(self.rawData)
    self.rawData = self.rawData.ravel()                      # Array 1D

    self.scaleFactor = self.load_struct_mat()['YInc']    # Factor de escala para convertir y[n] en voltios
    self.numSamples = self.load_struct_mat()['NumPoints']      # Número de muestras en la señal capturada (5e6 muestras)    
    self.Fs = int(20e9)               # Frecuencia de muestreo del osciloscopio (20 GSa/s).
    self.duration = self.numSamples/self.Fs              # Tiempo que abarca la señal capturada y[n] (0.25 ms)

    crop_start = int(2e6)
    crop_end = int(3e6)
    self.croppedData = self.rawData[crop_start:crop_end]     # Parte de los datos que incluye el transitorio de la señal para ahorrar almacenamiento
    self.croppedData = self.croppedData.ravel()              # Array 1D
    self.features = []    # Estructura que incluye las características extraídas

    if not args or args[0]=='OnlyRawData':
      self.croppedData = []
    elif args[0]=='OnlyCroppedData':
      self.rawData = []
    elif args[0]=='IncludeFeatures':
      #self.trajectory = self.get_energy_trayectory(plot=True)
      # De momento, features contiene sólo TransientStart pero puede contener más según sea necesario. 
      self.features = {'TransientStart': self.get_transient_start(self.croppedData)}     
    else:
      raise ValueError('Invalid Option!')

  

  def load_struct_mat(self):
    """ 
        Extrae el fichero .mat desde la memoria USB externa al directorio del proyecto
        y carga la estructura.
    """
    zip_path = r"D:\MPACT_DroneRC_RF_Dataset.zip"
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref: 
      source_path = rf"MPACT_DroneRC_RF_Dataset/{self.make}_{self.model}/{self.make}_{self.model}_{self.index}.mat"
      target_path = rf"D:\DataBase"     
      zip_ref.extract(source_path, target_path)

    extracted_mat_path = target_path + '\\' rf"MPACT_DroneRC_RF_Dataset\{self.make}_{self.model}\{self.make}_{self.model}_{self.index}.mat"
    mat = scipy.io.loadmat(extracted_mat_path)
    struct_mat = mat['data'][0,0]
    return struct_mat
  
  
  def get_energy_trayectory(self, plot=False):
    """
      Extracción de la trayectoria de máxima energía normalizada
      a partir de los datos reducidos (croppedData). 
    """
    # Convertimos los datos reducidos (croppedData a tipo float64) para poder calcular
    # el espectrograma
    data = self.croppedData.astype(np.float64)
    """
      Cálculo del espectrograma, basado en transformadas de fourier de ventanas sucesivas de la señal.
      
      Parámetros: 
          nperseg=128: La señal se divide en ventanas de 128 muestras. 
          noverlap=120: Número de muestras que se solapan entre ventanas. 
          nfft=256: Número de puntos usados para calcular la FFT. 
          scaling='density': Devuelve la densidad espectral de potencia (DSP).
          mode='complex': Cada valor de Sxx es un número complejo (amplitud y fase).
          window='hamming': Se usa ventana Hamming. 
    """    
    frec, time, Sxx = spectrogram(data, fs=self.Fs, window='hamming', nperseg=128,
                                  noverlap=120, nfft=256, scaling='density', mode='psd')
    """
      La trayectoria de la energía normalizada se puede obtener a partir del espectrograma a través 
      de los máximos valores de energía a lo largo del eje de tiempos.
      Entonces, con la función find_transient_start() estimaremos el transitorio de energía 
      buscando el cambio más abrupto en la media o la varianza de la trayectoria. 
    """
    # Calculamos el módulo del espectro
    magnitude = np.abs(Sxx)
    # Obtenemos el máximo de cada instante temporal (columna)
    max_per_time = np.max(magnitude, axis=0)
    # Normalizamos la trayectoria (norma L2)
    norm = np.linalg.norm(max_per_time)
    if norm != 0:
      trajectory = max_per_time / norm
    else:
      trajectory = max_per_time

    if plot: 
      # Gráfica de la trayectoria
      time_ms = time*1e3
      plt.plot(time_ms, trajectory)
      plt.xlabel('Time (ms)')
      plt.ylabel('Max Energy Trajectory')
      plt.title('Trayectoria de máxima energía en función del tiempo')
      plt.grid(True)
      plt.tight_layout()
      plt.savefig("energy_trajectory.png")
      # Gráfica del espectrograma
      """
      plt.figure(2)
      frec_MHz = frec * int(1e6)
      #Sxx_dB = 10*np.log10(Sxx) 
      plt.pcolormesh(time_ms, frec_MHz, Sxx, shading='gouraud')
      plt.xlabel('Time (ms)')
      plt.ylabel('Frecuency (MHz)')
      title = f"{self.make}_{self.model}_{self.index}"
      plt.title(title)
      plt.colorbar(label='PSD: Power Spectral Density (dB/Hz) ')
      plt.grid(True)
      plt.tight_layout()
      plt.savefig("spectrogram.png")
      """

    return trajectory
  
  def get_transient_start(self, cropData, plot=True):
    cropData = np.array(cropData)
    signal = hilbert(cropData) # Señal analítica (compleja) para extraer características (amplitud, fase)
    #s_i = np.real(s)      # Parte real (in-phase) de la señal s 
    #s_q = np.imag(s)      # Parte imaginaria (quadrature) de la señal s
    #amp = np.sqrt((s_i)**2+(s_q)**2) # Amplitud instantánea
    phi = np.angle(signal)        
    N = np.size(cropData)   # Tamaño de cropData
    av = np.unwrap(phi)     # Evita descontinuidades de fase.     
    window_size = 100                 # Tamaño de la ventana        
    # Vector temporal (TV) que almacena los valores de 'av' para cada ventana de tamaño 's'. Tamaño N/s
    tv = []
    num_windows = N//window_size      # Número total de ventanas no solapas en las que se divide 'av'
    for i in range(num_windows):
        g = (i+1)*window_size             # 'g' es el final de la ventana. (i+1) evitamos multiplicar por 0
        d = g - window_size               # 'd' marca el inicio de la porción de 'av' para calular la varianza
        window_i = av[d:g]
        tv.append(np.var(window_i))

    # Creamos la trayectoria fractal (FT)
    ft = np.abs(np.diff(tv))
    ft = np.array(ft)   
    ft_size = np.size(ft)
    transient_window = None
    for i in range(ft_size-4):
        if ft[i]<=5 and ft[i+1]<=5 and ft[i+2]<=5 and ft[i+3]<=5 and ft[i+4]<=5:
            transient_window = i
            break     # Sale del bucle cuando encuentra el primero
          
    transient_start = transient_window*window_size 

    if plot:
      time_ms = np.arange(np.size(self.croppedData)) / self.Fs * int(1e3)
      plt.plot(time_ms, self.croppedData*self.scaleFactor.ravel())
      plt.xlabel('Time (ms)')
      plt.ylabel('Amplitude (V)')
      title = f"{self.make}_{self.model}_{self.index}"
      plt.title(title)
      plt.grid(True)      
      plt.tight_layout()
      transient_start_ms = transient_start / self.Fs * 1e3          
      plt.axvline(transient_start_ms, color='r', linewidth=1.0, label='Transient Start')
      plt.savefig("transient_start.png")  
        
    return transient_start

"""  
  def find_transient_start_from_energy(self, trajectory, plot=False):  
    nperseg=128
    noverlap=120
    hop = nperseg-noverlap
    Lwnd  = 256
    Lslide = 10
    Lt = 20        
    hoc_vals = calc_hoc_values(trajectory, Lwnd=Lwnd, Lslide=Lslide)
    hoc_vals2 = hoc_vals**2
    T = []
    num_windows = np.size(hoc_vals)
    for i in range(num_windows - Lt):
        num_T = np.sum(hoc_vals2[i:i+Lt])
        den_T = np.sum(hoc_vals[i:i+Lt])**2       
        T.append(num_T/den_T)
    
    T = np.array(T)       
    transient_start = ((np.argmax(T) + 20)*Lslide + Lwnd)*hop + nperseg
    
    
    threshold = 0.0012*np.max(trajectory) # Umbral: 7% del valor máximo de energía        
    above_th = np.where(trajectory>threshold)[0] #     
    nperseg = 128
    noverlap = 120
    samples_non_overlapped = nperseg - noverlap  # samples_non_overlapped = 8 
    if np.size(above_th) != 0:   
      
        Para generar el espectrograma se divide la señal en ventanas de tamaño nperseg=128, 
        con un solapamiento entre ventanas adyacentes de noverlap=120. Es decir, cada índice 
        de 'trajectory' equivale a 8 muestras (samples_non_overlapped) de la señal original. 

        Por tanto, se realiza la operación: 
          transient_start = above_th[0]*samples_non_overlapped, siendo above_th[0] la ventana 
          donde se da el transitorio. 
      
    
      transient_start = above_th[0]*samples_non_overlapped
    else:
      transient_start = 0
    
    print("\nTransient_start", transient_start)


    if plot: 
      time_ms = np.arange(np.size(self.croppedData)) / self.Fs * int(1e3)
      plt.plot(time_ms, self.croppedData*self.scaleFactor.ravel())
      plt.xlabel('Time (ms)')
      plt.ylabel('Amplitude (V)')
      title = f"{self.make}_{self.model}_{self.index}"
      plt.title(title)
      plt.grid(True)      
      plt.tight_layout()
      transient_start_ms = transient_start / self.Fs * 1e3          
      plt.axvline(transient_start_ms, color='r', linewidth=1.0, label='Transient Start')
      plt.savefig("find_transient_start.png")          

    return transient_start


def calc_hoc(x):
    x_conj = np.conj(x)
    m42 = np.mean((x**2)*(x_conj**2))
    m20 = np.mean(x**2)
    m21 = np.mean(x*x_conj)
    c42 = m42 - np.abs(m20)**2 - 2*(m21**2) 
    hoc = np.sqrt(np.abs(c42))
    return hoc

def calc_hoc_values(x, Lwnd=256, Lslide=10):
    hoc_vals = []
    N = np.size(x)    
    for i in range(0, N-Lwnd+1, Lslide):
        window_i = x[i:i+Lwnd]
        hoc_i = calc_hoc(window_i)
        hoc_vals.append(hoc_i)

    print("\nTamaño de hoc_vals:", np.size(hoc_vals))    
    return np.array(hoc_vals)

def plot(self, option):
  
    Representa de forma gráfica el valor de 'option'

    Parámetro:
          option (str): 'RawData', 'CroppedData' o 'Spectrogram'
  
  opts = ['RawData', 'CroppedData', 'Spectrogram']
  if option not in opts:
    raise ValueError("Invalid plot option. \nValid options: ['RawData', 'CroppedData', 'Spectrogram']")
    
  if option == opts[0]: # Plot RawData
    time_ms = np.arange(np.size(self.rawData)) / self.Fs * int(1e3)
    plt.plot(time_ms, self.rawData*self.scaleFactor)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (V)')
    title = f"{self.make}_{self.model}_{self.index}"
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()    
  elif option == opts[1]: # Plot CroppedData
    time_ms = np.arange(np.size(self.croppedData)) / self.Fs * int(1e3)
    plt.plot(time_ms, self.croppedData*self.scaleFactor)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (V)')
    title = f"{self.make}_{self.model}_{self.index}"
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
  elif option == opts[2]: # Plot Spectrogram
    data = self.croppedData.astype(np.float64)
    frec, time, Sxx = spectrogram(data, fs=self.Fs, window='hamming', nperseg=128,
                                  noverlap=120, nfft=256, scaling='density', mode='complex')
    time_ms = time * int(1e3)
    frec_MHz = frec * int(1e6)
    Sxx_dB = 10*np.log10(Sxx) 
    plt.pcolormesh(time_ms, frec_MHz, Sxx_dB, shading='gouraud')
    plt.xlabel('Time (ms)')
    plt.ylabel('Frecuency (MHz)')
    title = f"{self.make}_{self.model}_{self.index}"
    plt.title(title)
    plt.colorbar(label='PSD: Power Spectral Density (dB/Hz) ')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
plt.close('all')

  # Plot raw data


  def find_transient_start(self, W, plot=False):
    
      Encuentra el inicio del transitorio de la señal (comienzo de la comunicación dron-controlador)
      analizando su complejidad (dimensión fractal) a lo largo del tiempo usando el método Higuchi.

      Parámetros: 
                  W: tamaño de la ventana temporal para calcular la dimensión fractal
                  plot: booleano para graficar el resultado (por defecto no se muestra)

      Devuelve: 
                  transientStart: muestra donde inicia el transitorio
    
    
    fd_arr = []                                             # Almacenará valores de la dimensión fractal para cada ventana
    data = self.croppedData.astype(np.float64).flatten()[:100000]    # Asegurar datos en 1D
    num_windows = np.size(data) // W                            # Número de ventanas
    for j in range(num_windows):
      window = data[W*j:W*(j+1)]
      fd_value = antropy.higuchi_fd(window, kmax=5)                          # Calcula la dimensión fractal usando el método Higuchi
      fd_arr.append(fd_value)
      
    fd_arr = np.array(fd_arr).reshape(-1, 1)

    
      Ahora detectamos el inicio del transitorio con un modelo de segmentación estadística.
      seg_model  = rpt.Pelt(model='rbf').fit(fd_arr)
          --> rpt.Pelt(...): Crea un objeto de detección de puntos usando el algoritmo
              PELT (Pruned Exact Linear Time). Busca los puntos en el tiempo donde las propiedades
              estadísticas de la señal cambian. 
          --> model = 'rbf': Usa una función de coste basada en el kernel RBF (Radial Basis Function) 
              para capturar cambios no lineales. Mide la NO similitud entre ventanas adyacentes. 
          --> .fit(fd_arr): Ajusta el modelo a los datos              
    
    seg_model  = rpt.Pelt(model='l2').fit(fd_arr)

    
        change_points = seg_model.predict(pen=1)
                --> predict(pen=1): Ejecuta el algoritmo para encontrar los puntos de cambio
                --> pen=1: Es la penalización por añadir un punto de cambio
                --> change_points: Lista de indices de ventanas donde el algoritmo detecta
                    cambios en la señal.

        Ejemplo: 
                change_windows = [12 43] significa cambios en las ventanas 12 y 43. 
    
    change_windows = seg_model.predict(pen=1)

    
        Cada elemento de fd_array representa la dimensión fractal de una ventana de W muestras. 
        Por tanto, si el cambio se produce en la ventana x, asumimos que el cambio se da en la última
        muestra de la ventana.
    
    if change_windows:
      transient_start = change_windows[0] * W
    else:
      transient_start = 0

    # Gráfica opcional
    if plot: 
      time_ms = np.arange(len(data)) / self.Fs * 1e3
      plt.figure(figsize=(8,4))
      plt.plot(time_ms, data, label="CroppedData")
      plt.axvline(x=transient_start / self.Fs * 1e3, color='r', linewidth=1.5, label='Transient Start')
      plt.xlabel("Time (ms)")
      plt.ylabel("Amplitude (V)")
      plt.title(f"{self.make}-{self.model}-{self.index}")
      plt.legend()
      plt.grid(True)
      plt.tight_layout()
      plt.show()

    return transient_start
  

  
  @staticmethod
  def Higuchi(x):
    
      Calcula la dimensión fractal (fd) de porción de señal.
      Por ejemplo, una señal del tipo [0.2, 0.1, 0.2, 0.2, 0.7, 0.9, 0.7, 0.8] representada de forma
      gráfica se vería como una línea quebrada. La idea es medir la curva si la recorremos cada 'k' 
      puntos empezando desde 'm'.
      Parámetros: 
                  x: porción de señal 
                  kmax: máximo valor de reducción (por defecto es 5)

      Devuelve: 
                  fd: dimensión fractal. Mide cuánto varía la señal.                     
    
    N = np.size(x)
    kmax=5
    Lmk = np.zeros((kmax, kmax))

    # Cálculo de la longitud media para cada valor de 'k' y 'm'
    for k in range(1, kmax+1):
      for m in range(1, k+1):
        Lmki = 0
        # num: Número de segmentos que podemos formar para empezando en 'm' y saltando cada 'k' puntos
        num = int(np.floor((N-m)/k))
        for i in range(1, num+1):
          Lmki += np.abs(x[m + i*k - 1] - x[m + (i - 1)*k - 1])

        if num > 0:
          # Ng es un factor de normalización para que la longitud sea comparable con distintos k
          Ng = (N-1)/(num*k)
        else:
          Ng = 0

        if k != 0 and num > 0:
          Lmk[m-1, k-1] = (Lmki*Ng)/k
        else:
          Lmk[m-1, k-1] = 0
        
        # Calculamos el promedio de longitudes para cada k        
        Lk = np.array([np.sum(Lmk[:k, k - 1]) / k 
                         if k > 0 else 0 for k in range(1, kmax + 1)
                         ])
                  
        # Evitamos la expresión log(0)
        valid = Lk > 0 
        lnLk = np.log(Lk[valid])
        lnk = np.log(1.0 / np.arange(1, kmax + 1)[valid])

        if len(lnk) >= 2: 
          # Se ajusta una recta a los puntos ln(1/k), ln(L(K)) usando mínimos cuadrados
          b = np.polyfit(lnk, lnLk, 1) 
          fd = b[0]            # La dimensión fractal es la pendiente de la recta estimada
        else:
          fd = 0               # La dimensión fractal es 0 si hay menos de 2 puntos
                  
                              
    return fd
"""
  


