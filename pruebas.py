from DronRC import DronRC
ruta = r"Z:\TFG\Bases de datos\RF-fingerprint\Data\MPACT_DroneRC_RF_Dataset.zip\MPACT_DroneRC_RF_Dataset\DJI_Inspire1Pro\DJI_Inspire1Pro_0001.mat"
dron = DronRC(ruta)
print(dron.datosSinProcesar[:5])
# Elimina archivo
#os.remove("C:/Users/joset/DroneRF_project/data/MPACT_DroneRC_RF_Dataset/DJI_Inspire1Pro/DJI_Inspire1Pro_0001.mat")

