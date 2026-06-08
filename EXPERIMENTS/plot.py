import sys
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'ieee'])

plt.rcParams.update({
    'xtick.labelsize': 21,
    'ytick.labelsize': 21,
    'legend.fontsize': 18,
    'axes.grid': True,
    'figure.figsize': (10, 6)
})

DATA_DIR = "../saves/"

def load_data():
    archivos = glob.glob(os.path.join(DATA_DIR, "*quadrupole*.npz"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos con 'quadrupole' en {DATA_DIR}")
    
    datos = {}
    for arch in archivos:
        match = re.search(r'Omega([\d\.]+)', arch)
        if match:
            omega_str = match.group(1).rstrip('.')
            try:
                Omega = float(omega_str)
            except ValueError:
                print(f"No se pudo convertir '{omega_str}' en {arch}")
                Omega = float(input(f"Valor de Omega para {os.path.basename(arch)}: "))
        else:
            print(f"Archivo sin patrón de Omega: {os.path.basename(arch)}")
            Omega = float(input("Introduce Omega: "))
        
        data = np.load(arch)
        datos[Omega] = {
            't': data['t'],
            'alpha': data['alpha'],
            'Lz': data['Lz']
        }
        print(f"Cargado Omega = {Omega}, t_max = {datos[Omega]['t'][-1]:.2f}, max|alpha| = {np.max(np.abs(datos[Omega]['alpha'])):.3f}")
    
    return datos

def plot_comparison(datos):
    omegas = sorted(datos.keys())
    colores = plt.cm.tab10(np.linspace(0, 1, len(omegas)))
    estilos = ['-', '--', '-.'] * (len(omegas)//4 + 1)

    plt.figure(figsize=(12, 6))
    for i, Omega in enumerate(omegas):
        d = datos[Omega]
        plt.plot(d['t'], d['alpha'], linestyle=estilos[i], label=rf'$\vartheta/\gamma = {Omega}$', linewidth=1.2)
    plt.xlabel(r'$\tau$', fontsize=21)
    plt.ylabel(r'$\alpha(t)$', fontsize=21)
    plt.legend()
    plt.tight_layout()
    plt.savefig("/home/hugo/Hugo_OMEN/TFG/GrAL/figuras/kuadrupoloa/quadrupole_comparison.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, 6))
    for i, Omega in enumerate(omegas):
        d = datos[Omega]
        plt.plot(d['t'], d['Lz'], linestyle=estilos[i], label=rf'$\vartheta/\gamma = {Omega}$', linewidth=1.2)
    plt.xlabel('Tiempo (ms)', fontsize=21)
    plt.ylabel(r'$\langle l_z \rangle$', fontsize=21)
    plt.legend()
    plt.tight_layout()
    plt.savefig("/home/hugo/Hugo_OMEN/TFG/GrAL/figuras/kuadrupoloa/Lz_comparison.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    datos = load_data()
    plot_comparison(datos)
    print("Todos los gráficos generados.")