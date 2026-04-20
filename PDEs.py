import numpy as np
from scipy.integrate import solve_ivp

def d2_neumann_axis0(A, dx):
    """Second derivative along axis=0 with Neumann (zero-flux) at both ends. But 1D array only."""
    left  = np.roll(A,  1, axis=0)
    right = np.roll(A, -1, axis=0)
    # impose ghost cells: u_{-1}=u_{1}, u_{N}=u_{N-2}
    left[0]  = A[1]
    right[-1] = A[-2]
    return (left + right - 2.0*A) / (dx*dx)

def reset_PDE(t,C, K, Nx, r, D, dx,):
    """
    Continuum limit PDEs for each cell stage and cell total under the reset
    model, in 1D. With some BC.

    :param t: Time, float
    :param C: Concentration of cells in each stage and site, array
    :param K: Number of stages, int
    :param Nx: Number of spatial points, int
    :param r: Proliferation rate, float
    :param D: Diffusion coefficient, float
    :param dx: Spatial step size, float

    Note C is passed in as a flattened array, so needs to be reshaped to (K+1, Nx).
    C[-1] corresponds to the concentration of total cells, while C[0] to C[K-1] correspond to 
    the concentrations of each stage.

    :return: dCdt, the derivative of C with respect to time, flattened to a 1D array.
    """

    #Reshape C to (K+1, Nx)
    C = np.reshape(C, (K+1, Nx))

    #Initialise derivative
    dCdt = np.zeros_like(C)

    ##ODE for stage 1 cells
    #Proliferation and progression term
    dCdt[0] += r*C[-2]*(2-C[-1]) - r*C[0]
    #Diffusion term x with second order finite difference and zero-flux boundaries
    dCdt[0] +=  (D *(1-C[-1])* d2_neumann_axis0(C[0], dx)) + (D * C[0] * d2_neumann_axis0(C[-1], dx))

    ##ODEs for stage 2 to K cells
    for k in range(1, len(C)-1):
        # Progression term
        dCdt[k] += r * C[k-1] - r * C[k]

        # Diffusion x-direction
        dCdt[k] += (D * (1 - C[-1]) * d2_neumann_axis0(C[k], dx)) + (D * C[k] * d2_neumann_axis0(C[-1], dx))
    
    ##ODE for total cells
    #Logisitc growth term
    dCdt[-1] += r * C[-2] * (1-C[-1])
    #Diffusion term x with second order finite difference and zero-flux boundaries
    dCdt[-1] += D*d2_neumann_axis0(C[-1], dx)

    #Flatten dCdt to 1D array
    return dCdt.flatten()

def myopic_reset_PDE(t,C, K, Nx, r, D, dx,):
    """
    Continuum limit PDEs for each cell stage and cell total under the reset
    model, in 1D. With some BC.

    :param t: Time, float
    :param C: Concentration of cells in each stage and site, array
    :param K: Number of stages, int
    :param Nx: Number of spatial points, int
    :param r: Proliferation rate, float
    :param D: Diffusion coefficient, float
    :param dx: Spatial step size, float

    Note C is passed in as a flattened array, so needs to be reshaped to (K+1, Nx).
    C[-1] corresponds to the concentration of total cells, while C[0] to C[K-1] correspond to 
    the concentrations of each stage.

    :return: dCdt, the derivative of C with respect to time, flattened to a 1D array.
    """

    #Reshape C to (K+1, Nx)
    C = np.reshape(C, (K+1, Nx))

    #Initialise derivative
    dCdt = np.zeros_like(C)

    ##ODE for stage 1 cells
    #Proliferation and progression term
    dCdt[0] += r*C[-2]*(2-(C[-1])**4) - r*C[0]
    #Diffusion term x with second order finite difference and zero-flux boundaries
    dCdt[0] +=  (D *(1-C[-1])* d2_neumann_axis0(C[0], dx)) + (D * C[0] * d2_neumann_axis0(C[-1], dx))

    ##ODEs for stage 2 to K cells
    for k in range(1, len(C)-1):
        # Progression term
        dCdt[k] += r * C[k-1] - r * C[k]

        # Diffusion x-direction
        dCdt[k] += (D * (1 - C[-1]) * d2_neumann_axis0(C[k], dx)) + (D * C[k] * d2_neumann_axis0(C[-1], dx))
    
    ##ODE for total cells
    #Logisitc growth term
    dCdt[-1] += r * C[-2] * (1-(C[-1])**4)
    #Diffusion term x with second order finite difference and zero-flux boundaries
    dCdt[-1] += D*d2_neumann_axis0(C[-1], dx)

    #Flatten dCdt to 1D array
    return dCdt.flatten()

def remain_PDE(t,C, K, Nx, r, D, dx,):
    """
    Continuum limit PDEs for each cell stage and cell total under the reset
    model, in 1D. With some BC.

    :param t: Time, float
    :param C: Concentration of cells in each stage and site, array
    :param K: Number of stages, int
    :param Nx: Number of spatial points, int
    :param r: Proliferation rate, float
    :param D: Diffusion coefficient, float
    :param dx: Spatial step size, float

    Note C is passed in as a flattened array, so needs to be reshaped to (K+1, Nx).
    C[-1] corresponds to the concentration of total cells, while C[0] to C[K-1] correspond to 
    the concentrations of each stage.

    :return: dCdt, the derivative of C with respect to time, flattened to a 1D array.
    """

    #Reshape C to (K+1, Nx)
    C = np.reshape(C, (K+1, Nx))

    #Initialise derivative
    dCdt = np.zeros_like(C)

    ##ODE for stage 1 cells
    #Proliferation and progression term
    dCdt[0] += 2*r*C[-2]*(1-C[-1]) - r*C[0]
    #Diffusion term x with second order finite difference and zero-flux boundaries
    dCdt[0] +=  (D *(1-C[-1])* d2_neumann_axis0(C[0], dx)) + (D * C[0] * d2_neumann_axis0(C[-1], dx))

    ##ODEs for stage 2 to K-1 cells
    for k in range(1, len(C)-2):
        # Progression term
        dCdt[k] += r * C[k-1] - r * C[k]

        # Diffusion x-direction
        dCdt[k] += (D * (1 - C[-1]) * d2_neumann_axis0(C[k], dx)) + (D * C[k] * d2_neumann_axis0(C[-1], dx))

    ##ODE for stage K cells
    # Proliferation term
    dCdt[-2] += r*C[-3] - r*C[-2]*(1-C[-1])

    #Diffusion term x with second order finite difference and zero-flux boundaries
    dCdt[-2] +=  (D *(1-C[-1])* d2_neumann_axis0(C[-2], dx)) + (D * C[-2] * d2_neumann_axis0(C[-1], dx))
    
    ##ODE for total cells
    #Logisitc growth term
    dCdt[-1] += r * C[-2] * (1-C[-1])
    #Diffusion term x with second order finite difference and zero-flux boundaries
    dCdt[-1] += D*d2_neumann_axis0(C[-1], dx)

    #Flatten dCdt to 1D array
    return dCdt.flatten()

def myopic_remain_PDE(t,C, K, Nx, r, D, dx,):
    """
    Continuum limit PDEs for each cell stage and cell total under the reset
    model, in 1D. With some BC.

    :param t: Time, float
    :param C: Concentration of cells in each stage and site, array
    :param K: Number of stages, int
    :param Nx: Number of spatial points, int
    :param r: Proliferation rate, float
    :param D: Diffusion coefficient, float
    :param dx: Spatial step size, float

    Note C is passed in as a flattened array, so needs to be reshaped to (K+1, Nx).
    C[-1] corresponds to the concentration of total cells, while C[0] to C[K-1] correspond to 
    the concentrations of each stage.

    :return: dCdt, the derivative of C with respect to time, flattened to a 1D array.
    """

    #Reshape C to (K+1, Nx)
    C = np.reshape(C, (K+1, Nx))

    #Initialise derivative
    dCdt = np.zeros_like(C)

    ##ODE for stage 1 cells
    #Proliferation and progression term
    dCdt[0] += 2*r*C[-2]*(1-(C[-1])**4) - r*C[0]
    #Diffusion term x with second order finite difference and zero-flux boundaries
    dCdt[0] +=  (D *(1-C[-1])* d2_neumann_axis0(C[0], dx)) + (D * C[0] * d2_neumann_axis0(C[-1], dx))

    ##ODEs for stage 2 to K-1 cells
    for k in range(1, len(C)-2):
        # Progression term
        dCdt[k] += r * C[k-1] - r * C[k]

        # Diffusion x-direction
        dCdt[k] += (D * (1 - C[-1]) * d2_neumann_axis0(C[k], dx)) + (D * C[k] * d2_neumann_axis0(C[-1], dx))

    ##ODE for stage K cells
    # Proliferation term
    dCdt[-2] += r*C[-3] - r*C[-2]*(1-(C[-1])**4)

    #Diffusion term x with second order finite difference and zero-flux boundaries
    dCdt[-2] +=  (D *(1-C[-1])* d2_neumann_axis0(C[-2], dx)) + (D * C[-2] * d2_neumann_axis0(C[-1], dx))
    
    ##ODE for total cells
    #Logisitc growth term
    dCdt[-1] += r * C[-2] * (1-(C[-1])**4)
    #Diffusion term x with second order finite difference and zero-flux boundaries
    dCdt[-1] += D*d2_neumann_axis0(C[-1], dx)

    #Flatten dCdt to 1D array
    return dCdt.flatten()

def PDE_solver(t_init, t_final, dt, C0, K, Nx, r, D, dx, PDE_func):
    """Solve a given PDE for a cell proliferation model given initial condition and parameter sets.
    :param t_init: Initial time, float
    :param t_final: Final time, float
    :param dt: Time step for simulation, float
    :param C0: Initial condition, array of shape (K+1, Nx)
    :param K: Number of stages, int
    :param Nx: Number of spatial points, int
    :param r: Proliferation rate, float
    :param D: Diffusion coefficient, float
    :param dx: Spatial step size, float
    :param PDE_func: The PDE function to solve, which should take in (t, C, K, Nx, r, D, dx) as arguments and return dCdt.
    :return: t, C where t is the array of time points and C is the array of concentrations at each time point, of shape (K+1, Nx, num_time_points).
    """
    num_steps = int((t_final - t_init) / dt)
    t_eval = np.linspace(t_init, t_final, num_steps)
    sol = solve_ivp(PDE_func, [t_init, t_final], C0.flatten(), args=(K, Nx, r, D, dx), t_eval=t_eval)
    return sol.t, sol.y