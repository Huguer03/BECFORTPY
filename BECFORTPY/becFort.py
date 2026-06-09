import numpy as np 
from scipy.fft import fft2, ifft2, fftfreq
from skimage.feature import blob_log
import gpe_solver

class Grid:
    def __init__(self, N, L):
        self.Nx = N[0]
        self.Ny = N[1]
        self.N  = N[0]*N[1]
        self.Lx = L[0]
        self.Ly = L[1]
        self.dx = self.Lx/self.Nx
        self.dy = self.Ly/self.Ny

        self.x = np.linspace(-self.Lx/2, self.Lx/2, self.Nx, endpoint=False)
        self.y = np.linspace(-self.Ly/2, self.Ly/2, self.Ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        self.kx = 2.0 * np.pi * fftfreq(self.Nx, self.dx)
        self.ky = 2.0 * np.pi * fftfreq(self.Ny, self.dy)
        self.Kx, self.Ky = np.meshgrid(self.kx, self.ky, indexing='ij')
        self.K2 = self.Kx**2 + self.Ky**2

        self.mesh_shape = (self.Nx, self.Ny)

    def fft(self, phi):
        return fft2(phi, norm="ortho")

    def ifft(self, phi):
        return ifft2(phi, norm="ortho")

    def laplacian(self, phi_k):
        return - self.K2 * phi_k

class WaveFunction:
    def __init__(self, grid, sigma=(1.0,1.0), gamma=(0,0), beta=0, Omega=0.0, phi=None):
        self.grid      = grid
        self.sigma_x   = sigma[0]
        self.sigma_y   = sigma[1]
        self.beta      = beta
        self.Omega     = Omega * min(gamma)
        if phi == None:
            phi = np.exp(-0.5 * ((self.grid.X / self.sigma_x)**2 + (self.grid.Y / self.sigma_y)**2), dtype=complex)

        self.phi = phi
        self.normalize()

    def normalize(self, A=1.0):
        norma = np.sum(np.abs(self.phi)**2) * self.grid.dx * self.grid.dy 
        self.phi *= np.sqrt(A / norma)

    def density(self):
        return np.abs(self.phi)**2 
    
    def phase(self):
        phase = np.angle(self.phi)
        return phase

    def norma(self):
        norma = np.sum(np.abs(self.phi)**2) * self.grid.dx * self.grid.dy
        return norma

    def energy(self):
        V      = self.potential(self.grid.X, self.grid.Y) 
        phi_k  = self.grid.fft(self.phi)
        pixel  = self.grid.dx * self.grid.dy
        E_kin  = 0.5 * np.sum(self.grid.laplacian(phi_k)) * pixel
        E_pot  = np.sum(V * np.abs(self.phi)**2) * pixel
        E_beta = self.beta * np.sum(np.abs(self.phi)**4) * pixel

        if self.Omega != 0.0:
            dx_phi = 1j * self.grid.Kx * phi_k
            dy_phi = 1j * self.grid.Ky * phi_k
            Lz_phi = -1j * (self.grid.X * dy_phi - self.grid.Y * dx_phi)
            E_rot  = -self.Omega * np.sum(np.conj(self.phi) * Lz_phi) * pixel
            return np.real(E_kin + E_pot + E_beta + E_rot)
        else:
            return np.real(E_kin + E_pot + E_beta)


class SSFM:
    def __init__(self, grid, beta=0):
        self.grid      = grid
        self.beta      = beta

    def evol(self, phi, final_time, dt, gamma, Omega, diss, W):
        if not diss:
            W = np.zeros_like(self.grid.X)
        phi_out = np.asfortranarray(phi.copy()).astype(np.complex128)
        kx      = np.asfortranarray(self.grid.Kx).astype(np.float64)
        ky      = np.asfortranarray(self.grid.Ky).astype(np.float64)
        k2      = np.asfortranarray(self.grid.K2).astype(np.float64)
        x       = np.asfortranarray(self.grid.X).astype(np.float64)
        y       = np.asfortranarray(self.grid.Y).astype(np.float64)
        w       = np.asfortranarray(W).astype(np.float64)

        gpe_solver.gpe_solver.ssfm_evol(
                            phi        = phi_out,
                            gx         = gamma[0],
                            gy         = gamma[1],
                            kx         = kx,
                            ky         = ky,
                            k2         = k2,
                            x          = x,
                            y          = y,
                            nx         = self.grid.Nx,
                            ny         = self.grid.Ny,
                            beta       = self.beta,
                            omega      = Omega,
                            final_time = final_time,
                            dt         = dt,
                            diss       = diss,
                            w          = w
                        )
        return phi_out
  
    def evolcool(self, phi, dt, n_vortex, vortex_charges, positions, tol, random_seed, max_iter, gamma, Omega):
        if n_vortex > 0: 
            phi = self.vortex_phase_mask(phi           = phi, 
                                        n_vortex       = n_vortex, 
                                        vortex_charges = vortex_charges, 
                                        positions      = positions, 
                                        random_seed    = random_seed
                                        )
            
        phi_out = np.asfortranarray(phi.copy()).astype(np.complex128)
        kx      = np.asfortranarray(self.grid.Kx).astype(np.float64)
        ky      = np.asfortranarray(self.grid.Ky).astype(np.float64)
        k2      = np.asfortranarray(self.grid.K2).astype(np.float64)
        x       = np.asfortranarray(self.grid.X).astype(np.float64)
        y       = np.asfortranarray(self.grid.Y).astype(np.float64)

        converge = gpe_solver.gpe_solver.imag_evol(
                                            phi      = phi_out,
                                            gx       = gamma[0],
                                            gy       = gamma[1],
                                            kx       = kx,
                                            ky       = ky,
                                            k2       = k2,
                                            x        = x,
                                            y        = y,
                                            nx       = self.grid.Nx,
                                            ny       = self.grid.Ny,
                                            dx       = self.grid.dx,
                                            dy       = self.grid.dy,
                                            beta     = self.beta,
                                            omega    = Omega,
                                            dt       = dt,
                                            max_iter = max_iter,
                                            tol      = tol
                                        )

        if converge == True:
            return phi_out
        else:
            raise ValueError("Maximun iterations reached, the whave function does not converge")
    
    def vortex_phase_mask(self, phi, n_vortex, vortex_charges, positions, random_seed):
        if n_vortex == 0:
            return phi
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        if vortex_charges is None:
            vortex_charges = [1] * n_vortex
        else:
            if len(vortex_charges) < n_vortex:
                raise TypeError("LESS vortex charges than vortex")
            elif len(vortex_charges) > n_vortex:
                raise TypeError("MORE vortex charges than vortex")
        
        if positions is None:
            max_radius = min(3.0, self.grid.Lx/4)
            
            positions = []
            for i in range(n_vortex):
                r     = np.random.uniform(0.5, max_radius)
                theta = np.random.uniform(0, 2*np.pi)
                x     = r * np.cos(theta)
                y     = r * np.sin(theta)
                positions.append((x, y))
        else:
            if len(positions) < n_vortex:
                raise TypeError("ERROR: LESS positions than vortex")
            elif len(positions) > n_vortex:
                raise TypeError("MORE positions than vortex")
        
        phi_with_vortices = phi.copy()

        for i, (charge, (x, y)) in enumerate(zip(vortex_charges, positions)):         
            phase = np.atan2(self.grid.Y - y, self.grid.X - x)
            
            vortex_phase = np.exp(1j * charge * phase)
            
            phi_with_vortices *= vortex_phase
        return phi_with_vortices

class ThomasFermi:
    def __init__(self, gamma, beta=0):
        self.mutf      = np.sqrt(beta / np.pi)
        self.rtf       = ((4.0 * beta) / (np.min(gamma)**2 * np.pi))**0.25
    
class Simulation:
    def __init__(self, grid, gamma, beta=0, Omega=0, n_vortex=0, vortex_charge=None, positions=None, sigma=(1.0,1.0), phi=None, seed=None):
        self.grid      = grid
        self.gamma     = gamma
        self.vortex    = n_vortex
        self.v_charge  = vortex_charge
        self.positions = positions
        self.beta      = beta
        self.Omega     = Omega * min(gamma)
        self.seed      = seed
        self.wf        = WaveFunction(grid, sigma, gamma, beta, Omega, phi)
        self.ssfm      = SSFM(grid, beta)

    def cooling(self, dt, tol=1E-13, max_iter=1000000, gamma=None, Omega=None):
        if gamma is None:
            gamma = self.gamma
        if Omega == None:
            Omega = self.Omega
        else:
            Omega *= np.min(gamma)
        if dt > 1/self.beta:
            raise ValueError("Time step discretization to lage")
        self.wf.phi = self.ssfm.evolcool(phi           = self.wf.phi,
                                        dt             = dt, 
                                        n_vortex       = self.vortex, 
                                        vortex_charges = self.v_charge, 
                                        positions      = self.positions, 
                                        tol            = tol, 
                                        random_seed    = self.seed,
                                        max_iter       = max_iter,
                                        gamma          = gamma,
                                        Omega          = Omega
                                        )
        self.wf.normalize()
    
    def hydrodynamics(self, t_max, dt, gamma=None, Omega=None,diss=False, W=None):
        if gamma is None:
            gamma = self.gamma
        if Omega is None:
            Omega = self.Omega
        else:
            Omega *= np.min(gamma)
        if dt > 1/self.beta:
            raise ValueError("Time step discretization to lage")
        self.wf.phi = self.ssfm.evol(phi        = self.wf.phi, 
                                     final_time = t_max, 
                                     dt         = dt,
                                     gamma      = gamma,
                                     Omega      = Omega,
                                     diss       = diss,
                                     W          = W
                                     )