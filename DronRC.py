# -*- coding: utf-8 -*-
import sys
assert sys.version_info >= (3, 7)
from packaging import version
import sklearn
from sklearn.model_selection import train_test_split
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.io import loadmat
from pathlib import Path
from scipy.signal import hilbert
from scipy.stats import skew, kurtosis, entropy

"""Definición de la clase DroneRC"""
class DronRC:
  def __init__(self, full_path, *args):
    self.full_path = Path(full_path)    
    self.file_name = self.full_path.parts[-1]
    if np.size(self.file_name.split('_')) == 3:                      
      self.make = self.file_name.split('_')[0]                       
      self.model = self.file_name.split('_')[1]                      
      self.index = self.file_name.split('_')[2].replace('.mat', '')  
    else:
      self.make = self.file_name.split('_')[0]                                          
      self.model = f"{self.file_name.split('_')[1]}_{self.file_name.split('_')[2]}"     
      self.index = self.file_name.split('_')[3].replace('.mat', '')  
    
    struct_mat = self.load_struct_mat()
    self.scaleFactor = struct_mat['YInc']
    self.numSamples = struct_mat['NumPoints']      
    self.Fs = int(20e9)                            
    self.duration = self.numSamples/self.Fs        
    self.rawData = struct_mat['Data']              
    self.rawData = self.rawData.astype(np.float64) - np.mean(self.rawData)
    self.rawData = np.array(self.rawData).ravel()  
    crop_start = int(2e6)
    crop_end = int(3e6)
    self.croppedData = self.rawData[crop_start:crop_end]     
    self.croppedData = np.array(self.croppedData).ravel()    
    if not args or args[0]=='OnlyRawData':
      self.croppedData = []
    elif args[0] == 'OnlyCroppedData':
      self.rawData = []
    elif args[0] == 'IncludeFeatures':
      self.rawData = []      
      self.transient_start = self.get_transient_start(self.croppedData)
      x = self.croppedData[self.transient_start::10]
      abs_x = np.abs(x)
      
      self.Mean = float(np.mean(x))                
      abs_mean = float(np.mean(abs_x))
      self.StandardDeviation = float(np.std(x))
      self.Skewness = float(skew(x))
      self.Entropy = float(entropy(x))
      self.RootMeanSquare = float(np.sqrt(np.mean(x**2)))
      self.Root = float(np.mean(np.sqrt(abs_x))**2)
      self.Kurtosis = float(kurtosis(x))
      self.Variance = float(np.var(x))
      self.Peak = float(np.max(x))
      self.Peak2Peak = float(np.max(x) - np.min(x))
      self.ShapeFactor = float(self.RootMeanSquare / abs_mean)
      self.CrestFactor = float(self.Peak / self.RootMeanSquare)
      self.ImpulseFactor = float(self.Peak / abs_mean)
      self.ClearanceFactor = float(self.Peak / self.Root)
    else:
      raise ValueError('Invalid Option!')

  def load_struct_mat(self):   
    mat = scipy.io.loadmat(self.full_path)
    struct_mat = mat['data'][0,0]
    return struct_mat
  
  def get_transient_start(self, cropData, plot=False):
    cropData = np.array(cropData)
    signal = hilbert(cropData)         
    phi = np.angle(signal)          
    N = np.size(cropData)           
    av = np.unwrap(phi)             
    window_size = 100                   
    tv = []
    num_windows = N//window_size    
    for i in range(num_windows):
        g = (i+1)*window_size        
        d = g - window_size          
        window_i = av[d:g]
        tv.append(np.var(window_i))
    
    ft = np.abs(np.diff(tv))
    ft = np.array(ft)       
    ft_size = np.size(ft)
    transient_window = None
    for i in range(ft_size-4):
        if ft[i]<=5 and ft[i+1]<=5 and ft[i+2]<=5 and ft[i+3]<=5 and ft[i+4]<=5:
            transient_window = i
            break     
          
    transient_start = transient_window*window_size 

    if plot:
      time_ms = np.arange(np.size(self.croppedData)) / self.Fs * int(1e3)
      plt.figure()
      plt.plot(time_ms, np.array(self.croppedData*self.scaleFactor).ravel())
      plt.xlabel('Time (ms)')
      plt.ylabel('Amplitude (V)')
      title = f"{self.make}_{self.model}_{self.index}"
      plt.title(title)
      plt.grid(True)      
      plt.tight_layout()
      transient_start_ms = transient_start / self.Fs * 1e3          
      plt.axvline(transient_start_ms, color='r', linewidth=1.0, label='Transient Start')
      plt.savefig("transient_start2.png")  
        
    return transient_start
  


