import sys
import os
sys.path.append(os.path.abspath('../BECFORTPY'))
import numpy as np
from becFort import Grid, TrapPotential, Simulation, ThomasFermi
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import utiles as ut
import scienceplots
plt.style.use(['science', 'ieee'])

plt.rcParams.update({
    'xtick.labelsize': 27,   
    'ytick.labelsize': 27,    
    'legend.fontsize': 23,    
    'axes.grid': True        
})

def linear(x, m, n):
    """
    Función lineal para el ajuste: y = m*x + n
    """
    return m * x + n

def coef_correlacion_lineal(x,y,popt):
    scr = np.sum((y - linear(x, *popt))**2)
    sct = np.sum((y - np.mean(y))**2)
    return 1 - scr/sct

def grafica(x,y,m,n,m_red,m_err,n_red,n_err,R,popt):
	plt.figure(figsize=(8, 6))
	leyenda_texto = (
	    f"Doiketa lineala:\n"
	    f"$m = ({m_red} \\pm {m_err})$\n"
	    f"$n = ({n_red} \\pm {n_err})$\n"
	    f"$R = {R:.4f}$"
	)
	plt.scatter(x, y, label='Simulazioen emaitzak', facecolors='none', color='black')
	plt.plot(x, linear(x, *popt), label=leyenda_texto, lw=1.5) 
	plt.xlabel(r'$\ln(\Delta \tau)$', size=30)
	plt.ylabel(r'$\ln(\text{Errorea}(\Delta \tau))$', size=30)
	plt.legend()
	plt.grid(True)
	plt.savefig(f'/home/hugo/Hugo_OMEN/TFG/GrAL/figuras/errore_ordena.png', dpi=300)

def programa():
    beta = 1000.0
    gamma = (10.0, 10.0)
    Omega = 0.0
    tf = ThomasFermi(gamma, beta)
    N = (2**6, 2**6)
    L = (8*tf.rtf, 8*tf.rtf)
    grid = Grid(N, L)
    print(tf.rtf, L)
    n_vortex = 0
    t=0.3
    tol = 1e-13

    sim = Simulation(grid          = grid, 
                     gamma         = gamma, 
                     beta          = beta, 
                     Omega         = Omega, 
                     n_vortex      = n_vortex, 
                     vortex_charge = None, 
                     positions     = None
                     )


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

    phi_0 = sim.wf.phi.copy()

    # Referencia
    phi_ref_ruta = f"../saves/phi{round(Omega,1)}_{round(gamma[0],1)}-{round(gamma[1],1)}_{n_vortex}_{t}_{N[0]}-{N[1]}_{round(L[0],3)}_{int(beta)}_ref.npy"
    if os.path.exists(phi_ref_ruta):
        print(f"Cargando estado de referencia desde {phi_ref_ruta}...")
        sim.wf.phi = np.load(phi_ref_ruta)
        print("Estado de referencia cargado.")
    else:
        print("No se ha encontrado estado referencia precargado.\nIniciando proceso de hidrodinamica...")
        sim.hydrodynamics(t,dt=1e-6)
        np.save(phi_ref_ruta, sim.wf.phi)
        print(f"Proceso finalizado. Nuevo estado de referencia guardado en {phi_ref_ruta}")
    
    phi_ref = sim.wf.phi.copy()

    dt_valores = np.array([5e-3, 2e-3, 1e-3, 5e-4, 3e-4])
    error_dt   = np.zeros(len(dt_valores))
    pixel      = sim.grid.dx * sim.grid.dy

    for i in range(len(dt_valores)):
    	print(dt_valores[i])
    	sim.wf.phi = phi_0.copy()
    	sim.hydrodynamics(t,dt=dt_valores[i])
    	phi_dt      = sim.wf.phi.copy()
    	error_dt[i] = np.sqrt(np.sum(np.abs(phi_ref-phi_dt)**2) * pixel)
    	print(error_dt[i])

    popt, pcov = curve_fit(linear,np.log(dt_valores),np.log(error_dt))
    m, n         = popt
    m_err, n_err = np.sqrt(np.diag(pcov))
    m_red, m_red_err = ut.redondear_escalares(m, m_err)
    n_red, n_red_err = ut.redondear_escalares(n, n_err)
    R = coef_correlacion_lineal(np.log(dt_valores),np.log(error_dt),popt)

    grafica(np.log(dt_valores),np.log(error_dt),m,n,m_red,m_red_err,n_red,n_red_err,R,popt)

if __name__ == "__main__":
	programa()