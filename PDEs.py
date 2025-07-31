import numpy as np
def continuum_reset_2D_params(t: float, C: np.ndarray, K: int, Nx: int, Ny: int, r: float, D: float, dx: float, dy: float):
    """
    Continuum limit PDEs for each cell stage and cell total under the reset
    model in 2D.

    :param t: Time (variable for ODE)
    :param C: Concentration of cells in each stage and site (variable for ODE)
    :param k: Number of stages
    :param Nx: Number of sites in x-direction
    :param Ny: Number of sites in y-direction
    :param r: Rate of cell proliferation
    :param D: Rate of motility
    :param dx: Grid spacing in x-direction
    :param dy: Grid spacing in y-direction

    Note C is passed in as a flattened array, so needs to be reshaped to (K+1, Nx, Ny).
    C[-1] corresponds to the concentration of total cells, while C[0] to C[K-1] correspond to 
    the concentrations of each stage.

    :return: dCdt, the derivative of C with respect to time, flattened to a 1D array.
    """

    #reshape C to (K+1, Nx, Ny)
    C = np.reshape(C, (K+1, Nx, Ny))

    #Initialise the derivative matrix
    dCdt = np.zeros_like(C)

    ##ODE for stage 1 cells
    #Proliferation and progression term
    dCdt[0] += r*C[-2]*(1-C[-1]) + r*C[-2] - r*C[0]
    #Diffusion term x with second order finite difference and zero-flux boundaries
    dCdt[0] += ( D *(1-C[-1])* (np.roll(C[0], 1, axis=0) + np.roll(C[0], -1, axis=0) +
                 - 2*C[0]) + D * C[0] * (np.roll(C[-1], 1, axis=0) + np.roll(C[-1], -1, axis=0) +
                 - 2*C[-1]) ) / (dx**2)
    #Diffusion term y with second order finite difference and zero-flux boundaries
    dCdt[0] += ( D * (1-C[-1]) *(np.roll(C[0], 1, axis=1) + np.roll(C[0], -1, axis=1) +
                 - 2*C[0]) + D * C[0] * (np.roll(C[-1], 1, axis=1) + np.roll(C[-1], -1, axis=1) +
                 - 2*C[-1]) )/ (dy**2)
    
    ##ODEs for stage 2 to K cells
    for k in range(1, len(C)-1):
        # Progression term
        dCdt[k] += r * C[k-1] - r * C[k]

        # Diffusion x-direction
        dCdt[k] += (D * (1 - C[-1]) * (np.roll(C[k], 1, axis=0) + np.roll(C[k], -1, axis=0) - 2*C[k]) +
                    D * C[k] * (np.roll(C[-1], 1, axis=0) + np.roll(C[-1], -1, axis=0) - 2*C[-1])) / dx**2

        # Diffusion y-direction
        dCdt[k] += (D * (1 - C[-1]) * (np.roll(C[k], 1, axis=1) + np.roll(C[k], -1, axis=1) - 2*C[k]) +
                    D * C[k] * (np.roll(C[-1], 1, axis=1) + np.roll(C[-1], -1, axis=1) - 2*C[-1])) / dy**2

    
    ##ODE for total cells
    #Logisitc growth term
    dCdt[-1] += r * C[-2] * (1-C[-1])
    #Diffusion term x with second order finite difference and zero-flux boundaries
    dCdt[-1] += (D * (np.roll(C[-1], 1, axis=0) + np.roll(C[-1], -1, axis=0) +
                 - 2*C[-1]) )/ (dx**2)
    #Diffusion term y with second order finite difference and zero-flux boundaries
    dCdt[-1] += (D * (np.roll(C[-1], 1, axis=1) + np.roll(C[-1], -1, axis=1) +
                 - 2*C[-1]) )/ (dy**2)
    
    #Flatten the derivative matrix to return
    return dCdt.flatten()

def continuum_remain_2D_params(t: float, C: np.ndarray, K: int, Nx: int, Ny: int, r: float, D: float, dx: float, dy: float):
    """
    Continuum limit PDEs for each cell stage and cell total under the remain
    model, but in 2D.

    :param t: Time
    :param C: Concentration of cells in each stage and site
    :param k: Number of stages
    :param Nx: Number of sites in x-direction
    :param Ny: Number of sites in y-direction
    :param r: Rate of cell proliferation
    :param D: Rate of motility
    :param dx: Grid spacing in x-direction
    :param dy: Grid spacing in y-direction

    Note C is passed in as a flattened array, so needs to be reshaped to (K+1, Nx, Ny).
    C[-1] corresponds to the concentration of total cells, while C[0] to C[K-1] correspond to 
    the concentrations of each stage.

    :return: dCdt, the derivative of C with respect to time, flattened to a 1D array.
    """

    #reshape C to (K+1, Nx, Ny)
    C = np.reshape(C, (K+1, Nx, Ny))

    #Initialise the derivative matrix
    dCdt = np.zeros_like(C)

    ##ODE for stage 1 cells
    #Proliferation and progression term
    dCdt[0] += 2*r*C[-2]*(1-C[-1]) - r*C[0]
    #Diffusion term x with second order finite difference and zero-flux boundaries
    dCdt[0] += ( D *(1-C[-1])* (np.roll(C[0], 1, axis=0) + np.roll(C[0], -1, axis=0) +
                 - 2*C[0]) + D * C[0] * (np.roll(C[-1], 1, axis=0) + np.roll(C[-1], -1, axis=0) +
                 - 2*C[-1]) ) / (dx**2)
    #Diffusion term y with second order finite difference and zero-flux boundaries
    dCdt[0] += ( D * (1-C[-1]) *(np.roll(C[0], 1, axis=1) + np.roll(C[0], -1, axis=1) +
                 - 2*C[0]) + D * C[0] * (np.roll(C[-1], 1, axis=1) + np.roll(C[-1], -1, axis=1) +
                 - 2*C[-1]) )/ (dy**2)
    
    ##ODEs for stage 2 to k cells
    for k in range(1, len(C)-2):
        # Progression term
        dCdt[k] += r * C[k-1] - r * C[k]

        # Diffusion x-direction
        dCdt[k] += (D * (1 - C[-1]) * (np.roll(C[k], 1, axis=0) + np.roll(C[k], -1, axis=0) - 2*C[k]) +
                    D * C[k] * (np.roll(C[-1], 1, axis=0) + np.roll(C[-1], -1, axis=0) - 2*C[-1])) / dx**2

        # Diffusion y-direction
        dCdt[k] += (D * (1 - C[-1]) * (np.roll(C[k], 1, axis=1) + np.roll(C[k], -1, axis=1) - 2*C[k]) +
                    D * C[k] * (np.roll(C[-1], 1, axis=1) + np.roll(C[-1], -1, axis=1) - 2*C[-1])) / dy**2
    
    ##ODEs for stage K cells
    # Progression term
    dCdt[-2] += r * C[-3] - r * C[-2] * (1-C[-1])

    #Diffusion x-direction
    dCdt[-2] += (D * (1 - C[-1]) * (np.roll(C[-2], 1, axis=0) + np.roll(C[-2], -1, axis=0) - 2*C[-2]) +
                 D * C[-2] * (np.roll(C[-1], 1, axis=0) + np.roll(C[-1], -1, axis=0) - 2*C[-1]) ) / dx**2
    #Diffusion y-direction
    dCdt[-2] += (D * (1 - C[-1]) * (np.roll(C[-2], 1, axis=1) + np.roll(C[-2], -1, axis=1) - 2*C[-2]) +
                 D * C[-2] * (np.roll(C[-1], 1, axis=1) + np.roll(C[-1], -1, axis=1) - 2*C[-1])) / dy**2

    
    ##ODE for total cells
    #Logisitc growth term
    dCdt[-1] += r * C[-2] * (1-C[-1])
    #Diffusion term x with second order finite difference and zero-flux boundaries
    dCdt[-1] += (D * (np.roll(C[-1], 1, axis=0) + np.roll(C[-1], -1, axis=0) +
                 - 2*C[-1]) )/ (dx**2)
    #Diffusion term y with second order finite difference and zero-flux boundaries
    dCdt[-1] += (D * (np.roll(C[-1], 1, axis=1) + np.roll(C[-1], -1, axis=1) +
                 - 2*C[-1]) )/ (dy**2)
    
    #Flatten the derivative matrix to return
    return dCdt.flatten()
