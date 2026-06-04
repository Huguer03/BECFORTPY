import sys
import os
sys.path.append(os.path.abspath('../BECFORTPY'))
import numpy as np
import matplotlib.pyplot as plt
from becFort import Grid, TrapPotential, Simulation, ThomasFermi
import scienceplots
plt.style.use(['science', 'ieee'])

plt.rcParams.update({
    'xtick.labelsize': 25,   
    'ytick.labelsize': 25,    
    'legend.fontsize': 27,    
    'axes.grid': True        
})

def graf_energia(t,energia,Omega,dt):
    plt.figure(figsize=(8, 6))
    plt.plot(t, energia, lw=1.5) 
    plt.xlabel(r'$\tau$', size=27)
    plt.ylabel(r'$\Delta E(\tau)$', size=27)
    plt.grid(True)
    plt.savefig(f'/home/hugo/Hugo_OMEN/TFG/GrAL/figuras/energia_{str(Omega).replace('.', '-')}_10.png', dpi=300)

def graf_norma(t,norma,Omega,dt):
    plt.figure(figsize=(8, 6))
    plt.plot(t, norma[0], lw=1.5, label=r'(a)$\vartheta/\gamma = 0.0$') 
    plt.plot(t, norma[1], lw=1.5, label=r'(b)$\vartheta/\gamma = 0.5$')
    plt.plot(t, norma[2], lw=1.5, label=r'(c)$\vartheta/\gamma = 0.8$')
    plt.xlabel(r'$\tau$', size=30)
    plt.ylabel(r'$\|\phi\|_{L^2}(\tau)-1$', size=30)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'/home/hugo/Hugo_OMEN/TFG/GrAL/figuras/norma_10.png', dpi=300)

def test(w=0.0, dt=1.0):
    beta = 1000.0    
    gamma = (10.0, 10.0)
    Omega = w
    tf = ThomasFermi(gamma, beta)
    N = (2**8, 2**8)
    L = (10*tf.rtf, 10*tf.rtf)
    grid = Grid(N, L)
    print(tf.rtf, L, L[0]/N[0])
    n_vortex = 0
    tol = 1e-13

    sim = Simulation(grid          = grid, 
                     gamma         = gamma, 
                     beta          = beta, 
                     Omega         = Omega, 
                     n_vortex      = 0, 
                     vortex_charge = None, 
                     positions     = None
                     )
    
    # 4. Ejecutar el cooling
    phi_0_ruta = f"../saves/phi{round(Omega,1)}_{round(gamma[0],1)}-{round(gamma[1],1)}_{n_vortex}_{tol:.0e}_{N[0]}-{N[1]}_{round(L[0],3)}_{int(beta)}.npy"
    if os.path.exists(phi_0_ruta):
        print(f"Cargando estado fundamental desde {phi_0_ruta}...")
        sim.wf.phi = np.load(phi_0_ruta)
        print("Estado fundamental cargado.")
    else:
        print("No se ha encontrado estado fundamental precargado.\nIniciando proceso de cooling (Gradient descent)...")
        sim.cooling(1e-4, tol=tol)
        np.save(phi_0_ruta, sim.wf.phi)
        print(f"Cooling finalizado. Nuevo estado fundamental guardado en {phi_0_ruta}")

    t_max   = dt
    t       = np.linspace(0,t_max,500)
    delta_t = t_max / 500
    energia = [0]
    norma = [0]

    print("Cooling finalizado.")
    E_0 = sim.wf.energy()
    print(0, norma[0], energia[0])

    # 5. Vamos a simular la hidrodinamica
    for i in range(len(t)-1):
        sim.hydrodynamics(delta_t,dt=1e-3)
        norma.append(sim.wf.norma()-1)
        energia.append(sim.wf.energy()-E_0)
        print(i+1, norma[i+1], energia[i+1])

    # 6. Visualización de resultados

    #if Omega in [0.0,0.5,0.8]:
        #graf_energia(t,energia,Omega,dt)
        #graf_norma(t,norma,Omega,dt)

    return norma

if __name__ == "__main__":
    w = np.linspace(0,0.9,10)
    w = [0.0,0.5,0.8]
    print(w)
    dt = 10
    norma = []
    t = np.linspace(0,dt,500)
    for Omega in w:
        norma.append(test(Omega,dt))
    graf_norma(t,norma,w,dt)