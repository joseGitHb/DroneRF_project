import numpy as np
import matplotlib.pyplot as plt
def transient_start(x): 
    Lwnd  = 256
    Lslide = 10
    Lt = 20        
    hoc_vals = calc_hoc_values(x, Lwnd=Lwnd, Lslide=Lslide)
    hoc_vals2 = hoc_vals**2
    T = []
    num_windows = np.size(hoc_vals)
    for i in range(num_windows - Lt):
        num_T = np.sum(hoc_vals2[i:i+Lt])
        den_T = np.sum(hoc_vals[i:i+Lt])**2       
        T.append(num_T/den_T)
    
    T = np.array(T)
    print("Tamaño de T:", np.size(T))
    plt.plot(np.arange(num_windows-Lt), T)
    plt.title("RSM")
    plt.xlabel("Ventana i-ésima")
    plt.ylabel("T(i)")
    plt.savefig("RSM.png")
    transient_start = ((np.argmax(T) + 20)*Lslide + Lwnd)*8 + 128
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
    


