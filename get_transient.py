import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
def get_transient(cropData):
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



    plt.plot(np.arange(np.size(ft)), ft)
    plt.grid(True)
    plt.axvline(transient_window, color='r', linewidth=1.0, label='Transient Window')
    plt.savefig("Fractal Trajectory.png")
    print("Transient window:", transient_window)
    transient_start = transient_window*window_size 
    

    return transient_start


