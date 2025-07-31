##Solving PDEs using the method of lines and returning the solution array
import numpy as np
from scipy.integrate import solve_ivp
from PDEs import continuum_reset_2D_params
from PDEs import continuum_remain_2D_params
from PDEs import continuum_myopic_2D_params

def solve_reset_PDE(C0: np.ndarray, t_span: tuple, dt: float, K: int, Nx: int, Ny: int, r: float, D: float, dx: float, dy: float):
    """
    Solve the PDEs for the reset model using the method of lines.

    :param C0: Initial concentration of cells in each stage and site, flattened array
    :param t_span: Time span for the simulation as a tuple (t_init, t_final)
    :param dt: Time step for simulation
    :param K: Number of stages
    :param Nx: Number of sites in x-direction
    :param Ny: Number of sites in y-direction
    :param r: Rate of cell proliferation
    :param D: Rate of motility
    :param dx: Grid spacing in x-direction
    :param dy: Grid spacing in y-direction
    :return: Solution array with shape (num_timesteps+1, (K+1)*Nx*Ny)
    """
    
    # Define the ODE system using the PDE function
    def continuum_reset_2D(t, C):
        return continuum_reset_2D_params(t, C, K, Nx, Ny, r, D, dx, dy)

    #Define the time points to solve the ODE system
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

    # Solve the ODE system using solve_ivp
    sol = solve_ivp(continuum_reset_2D, t_span, C0.flatten(), t_eval=t_eval, method='RK45')

    # Reshape so that solution is shape (K,Nx,Ny,time_steps)
    sol_C = sol.y #Get solution array
    sol_C = sol_C.reshape((K+1, Nx, Ny, -1))

    return sol_C

def solve_remain_PDE(C0: np.ndarray, t_span: tuple, dt: float, K: int, Nx: int, Ny: int, r: float, D: float, dx: float, dy: float):
    """
    Solve the PDEs for the remain model using the method of lines.

    :param C0: Initial concentration of cells in each stage and site, flattened array
    :param t_span: Time span for the simulation as a tuple (t_init, t_final)
    :param dt: Time step for simulation
    :param K: Number of stages
    :param Nx: Number of sites in x-direction
    :param Ny: Number of sites in y-direction
    :param r: Rate of cell proliferation
    :param D: Rate of motility
    :param dx: Grid spacing in x-direction
    :param dy: Grid spacing in y-direction
    :return: Solution array with shape (num_timesteps+1, (K+1)*Nx*Ny)
    """
    
    # Define the ODE system using the PDE function
    def continuum_remain_2D(t, C):
        return continuum_remain_2D_params(t, C, K, Nx, Ny, r, D, dx, dy)

    #Define the time points to solve the ODE system
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

    # Solve the ODE system using solve_ivp
    sol = solve_ivp(continuum_remain_2D, t_span, C0.flatten(), t_eval=t_eval, method='RK45')

    # Reshape so that solution is shape (K,Nx,Ny,time_steps)
    sol_C = sol.y #Get solution array
    sol_C = sol_C.reshape((K+1, Nx, Ny, -1))

    return sol_C

def solve_myopic_PDE(C0: np.ndarray, t_span: tuple, dt: float, K: int, Nx: int, Ny: int, r: float, D: float, dx: float, dy: float):
    """
    Solve the PDEs for the myopic model using the method of lines.

    :param C0: Initial concentration of cells in each stage and site, flattened array
    :param t_span: Time span for the simulation as a tuple (t_init, t_final)
    :param dt: Time step for simulation
    :param K: Number of stages
    :param Nx: Number of sites in x-direction
    :param Ny: Number of sites in y-direction
    :param r: Rate of cell proliferation
    :param D: Rate of motility
    :param dx: Grid spacing in x-direction
    :param dy: Grid spacing in y-direction
    :return: Solution array with shape (num_timesteps+1, (K+1)*Nx*Ny)
    """
    
    # Define the ODE system using the PDE function
    def continuum_myopic_2D(t, C):
        return continuum_myopic_2D_params(t, C, K, Nx, Ny, r, D, dx, dy)

    #Define the time points to solve the ODE system
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

    # Solve the ODE system using solve_ivp
    sol = solve_ivp(continuum_myopic_2D, t_span, C0.flatten(), t_eval=t_eval, method='RK45')

    # Reshape so that solution is shape (K,Nx,Ny,time_steps)
    sol_C = sol.y #Get solution array
    sol_C = sol_C.reshape((K+1, Nx, Ny, -1))

    return sol_C
