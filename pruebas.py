from DronRC import DronRC
import numpy as np
from scipy.signal import spectrogram, hilbert
import matplotlib.pyplot as plt
#ruta = r"D:\MPACT_DroneRC_RF_Dataset.zip\MPACT_DroneRC_RF_Dataset\Futaba_T8FG\Futaba_T8FG_0009.mat"
#ruta = r"D:\MPACT_DroneRC_RF_Dataset.zip\MPACT_DroneRC_RF_Dataset\DJI_Phantom3\DJI_Phantom3_0001.mat"
#ruta = r"D:\MPACT_DroneRC_RF_Dataset.zip\MPACT_DroneRC_RF_Dataset\Graupner_MC32\Graupner_MC32_0004.mat"
#ruta = r"D:\MPACT_DroneRC_RF_Dataset.zip\MPACT_DroneRC_RF_Dataset\DJI_Matrice100\DJI_Matrice100_0001.mat"
#ruta = r"D:\MPACT_DroneRC_RF_Dataset.zip\MPACT_DroneRC_RF_Dataset\DJI_Matrice600_2\DJI_Matrice600__0001.mat"
#ruta = r"D:\MPACT_DroneRC_RF_Dataset.zip\MPACT_DroneRC_RF_Dataset\Turnigy_9X\Turnigy_9X_0001.mat"
#ruta = r"D:\MPACT_DroneRC_RF_Dataset.zip\MPACT_DroneRC_RF_Dataset\Spektrum_DX6i\Spektrum_DX6i_0001.mat"
#ruta = r"D:\MPACT_DroneRC_RF_Dataset.zip\MPACT_DroneRC_RF_Dataset\FlySky_FST6\FlySky_FST6_0001.mat"
ruta = r"D:\MPACT_DroneRC_RF_Dataset.zip\MPACT_DroneRC_RF_Dataset\DJI_Inspire1Pro_\DJI_Inspire1Pro_0001.mat"
# Drones a quitar 
#ruta = r"D:\MPACT_DroneRC_RF_Dataset.zip\MPACT_DroneRC_RF_Dataset\DJI_Phantom4Pro_1\DJI_Phantom4Pro_2_0001.mat"
#ruta = r"D:\MPACT_DroneRC_RF_Dataset.zip\MPACT_DroneRC_RF_Dataset\HobbyKing_HKT6A\HobbyKing_HKT6A_0001.mat"
#ruta = r"D:\MPACT_DroneRC_RF_Dataset.zip\MPACT_DroneRC_RF_Dataset\JetiDuplex_DC16\JetiDuplex_DC16_0001.mat"
#ruta = r"D:\MPACT_DroneRC_RF_Dataset.zip\MPACT_DroneRC_RF_Dataset\Spektrum_JRX9303\Spektrum_JRX9303_0001.mat"





from transient_start import transient_start
from transient_start import calc_hoc_values
dron = DronRC(ruta, 'IncludeFeatures')
#print("Tamaño croppedData:", np.size(dron.croppedData))
#trajectory = dron.get_energy_trayectory(plot=True)
#from get_transient import get_transient
#transient_start = get_transient(dron.croppedData)
#transient_start = dron.features['TransientStart']
#print("Transient start:", transient_start)

#print("Tamaño de trajectory:", np.size(trajectory))
#ts = transient_start(trajectory)
#print("Transient start: ", ts)
#hoc_vals = calc_hoc_values(trajectory)
#print(np.size(hoc_vals))
#print(np.size(trajectory))

#print(np.mean(trajectory[:5000]))
#transientStart = dron.find_transient_start_from_energy(trajectory, plot=True)

"""
data = dron.croppedData.astype(np.float64)
print(np.size(data))
frec, time, Sxx = spectrogram(data, fs=dron.Fs, window='hamming', nperseg=128,
                                  noverlap=120, nfft=256, scaling='density', mode='complex')
frec_MHz = frec * int(1e6)
time_ms = time* int(1e3)
print(np.mean(np.abs(Sxx)))
Sxx_dB = 10*np.log10(Sxx) 

plt.pcolormesh(time_ms, frec_MHz, 10*np.log10(Sxx), shading='gouraud')
plt.xlabel('Time (ms)')
plt.ylabel('Frecuency (MHz)')
title = f"{dron.make}_{dron.model}_{dron.index}"
plt.title(title)
plt.colorbar(label='PSD: Power Spectral Density (dB/Hz) ')
plt.grid(True)
plt.tight_layout()
plt.savefig("espectrograma.png", dpi=150)
"""
#plt.show()

#W = 50
#t = dron.find_transient_start(W, plot=True)
#trajectory = dron.get_energy_trayectory(plot=True)
#transient_start = dron.find_transient_start_from_energy(trajectory, plot=True)

#print('\n', dron.rawData[:5])
#print('\n', dron.scaleFactor)
#print('\n', dron.croppedData.shape)
#print('\n', dron.numSamples)
#print('\n', dro n.duration)


# Elimina archivo
#os.remove("C:\Users\joset\DroneRF_project/data/MPACT_DroneRC_RF_Dataset/DJI_Inspire1Pro/DJI_Inspire1Pro_0001.mat")
from createDataBase import createDataBase
zipPath = r"D:\MPACT_DroneRC_RF_Dataset.zip"
numSignals = 1
opts = 'OnlyCroppedData'
tableFormat = True
dataBase = createDataBase(zipPath, numSignals, opts, tableFormat)
print(dataBase[['make', 'model', 'index', 'croppedData', 'scaleFactor', 'numSamples' ]])

