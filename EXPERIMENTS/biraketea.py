import sys
import os
sys.path.append(os.path.abspath('../BECFORTPY'))
import numpy as np
import matplotlib.pyplot as plt
from becFort import Grid, Simulation, ThomasFermi
import scienceplots
plt.style.use(['science', 'ieee'])

plt.rcParams.update({
    'xtick.labelsize': 21,   
    'ytick.labelsize': 21,    
    'legend.fontsize': 23,    
    'axes.grid': False        
})

def quadrupole(phi, grid, Omega):
    rho = np.abs(phi)**2
    norm = np.sum(rho) * grid.dx * grid.dy
    
    x2 = np.sum(rho * grid.X**2) * grid.dx * grid.dy / norm
    y2 = np.sum(rho * grid.Y**2) * grid.dx * grid.dy / norm
    xy = np.sum(rho * grid.X * grid.Y) * grid.dx * grid.dy / norm
    
    alpha = -Omega * (x2 - y2) / (x2 + y2 + 1e-12) 
    return alpha, x2, y2, xy

def angular_momentum(phi, grid):
    phi_k = grid.fft(phi)
    dx_phi = grid.ifft(1j * grid.Kx * phi_k)
    dy_phi = grid.ifft(1j * grid.Ky * phi_k)
    Lz_op = -1j * (grid.X * dy_phi - grid.Y * dx_phi)
    Lz = np.sum(np.conj(phi) * Lz_op) * grid.dx * grid.dy
    norm = np.sum(np.abs(phi)**2) * grid.dx * grid.dy
    return np.real(Lz / norm)

def graf_densidad(L, density, t, Omega):
    t = round(t,3)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6)) 
    zoom_region = [-L[0]/6, L[0]/6, -L[0]/6, L[0]/6]
    im = ax.imshow(density, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2], cmap='inferno')
    ax.set_title(rf"$\Omega={Omega}$, $\tau = {t}$", size=23)
    ax.axis(zoom_region)
    ax.set_xticks([])   
    ax.set_yticks([])
    fig.colorbar(im, ax=ax)
    plt.savefig(f'../saves/simulaciones/densidad_{t}.png', dpi=300)
    plt.close()

def graf_phase(L, phase, t, Omega):
    t = round(t,3)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6)) 
    zoom_region = [-L[0]/6, L[0]/6, -L[0]/6, L[0]/6]
    im = ax.imshow(phase, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2], cmap='Greys')
    ax.set_title(rf"$\Omega={Omega}$, $\tau = {t}$", size=23)
    ax.axis(zoom_region)
    ax.set_xticks([])   
    ax.set_yticks([])
    fig.colorbar(im, ax=ax)
    plt.savefig(f'../saves/simulaciones/phase_{t}.png', dpi=300)
    plt.close()

def run_simulation(Omega, gy_final=9.9, ramp_time=0.2, t_max=10.0, dt=1e-3, save_images=False):
    beta = 500.0    
    gamma0 = (1.0, 1.0)  
    sigma = (1.0, 1.0)
    tf = ThomasFermi(gamma0, beta)
    N = (2**8, 2**8)
    L = (10*tf.rtf, 10*tf.rtf)
    grid = Grid(N, L)
    print(f"Omega={Omega}: tf.rtf={tf.rtf}, L={L}, dx={L[0]/N[0]}")
    n_vortex = 0
    tol = 1e-13
    vortex_charge = [1]
    positions = [(0.0, 0.0)]
    
    sim = Simulation(grid          = grid, 
                     gamma         = gamma0, 
                     sigma         = sigma, 
                     beta          = beta, 
                     Omega         = 0,
                     n_vortex      = n_vortex, 
                     vortex_charge = vortex_charge, 
                     positions     = positions
                     )

    
    phi_0_ruta = f"../saves/phi0.0_{gamma0[0]}-{gamma0[1]}_{sigma[0]}-{sigma[1]}_{n_vortex}_{tol:.0e}_{N[0]}-{N[1]}_{round(L[0],3)}_{int(beta)}.npy"
    if os.path.exists(phi_0_ruta):
        print(f"Cargando estado fundamental desde {phi_0_ruta}...")
        sim.wf.phi = np.load(phi_0_ruta)
        print("Estado fundamental cargado.")
    else:
        print("No se ha encontrado estado fundamental precargado.\nIniciando proceso de cooling...")
        sim.cooling(1e-4, tol=tol, max_iter=20000000)
        np.save(phi_0_ruta, sim.wf.phi)
        print(f"Cooling finalizado. Estado guardado en {phi_0_ruta}")
    
    if save_images:
        graf_densidad(L, sim.wf.density(), 0, Omega)
        graf_phase(L, sim.wf.phase(), 0, Omega)
    
    t_vals = []
    alpha_vals = []
    Lz_vals = []
    
    time = 0.0
    steps = int(t_max / dt)

    w = np.full_like(grid.X,0.04)
    
    for step in range(steps):
        time += dt
        
        if time < ramp_time:
            gy = 1.0 + (gy_final - 1.0) * (time / ramp_time)
        else:
            gy = gy_final
        
        sim.hydrodynamics(t_max=dt, dt=dt, gamma=(1.0, gy), Omega=Omega)
        
        if save_images and (step % 1000 == 0):
            graf_densidad(L, sim.wf.density(), time, Omega)
            graf_phase(L, sim.wf.phase(), time, Omega)

    data_filename = f"../saves/quadrupole_Omega{round(Omega,3)}.npz"
    np.savez(data_filename, t=t_vals, alpha=alpha_vals, Lz=Lz_vals, Omega=Omega, gy_final=gy_final)
    print(f"Datos guardados en {data_filename}")
    return t_vals, alpha_vals, Lz_vals

def main():
    Omega_list = [0.85]
    gy_final = 1.1
    ramp_time = 1.0
    t_max = 100.0
    dt = 1e-3
    save_images = True 
    
    for Omega in Omega_list:
        print(f"Simulando Omega = {Omega}")
        run_simulation(Omega, gy_final, ramp_time, t_max, dt, save_images)

if __name__ == "__main__":
    main()