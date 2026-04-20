import numpy as np

def wavespeed(array, dt):
    """Calculate average wave-speed per unit timestep by taking the average mass added.
    :param array: Array of concentration values (num_timesteps, Nx) 
    :param dt: Time step size, float
    :return: Average wave-speed (num_timesteps)
    """

    #Change in mass at each spatial point between time steps
    mass_change = np.diff(array, axis=0)

    #Sum across spatial dimension to get total mass added at each time step
    mass_added = np.sum(mass_change, axis=1) 

    #Average wave-speed per unit time step
    wave_speed = mass_added / dt  

    return wave_speed

def moving_average(wavespeeds: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute the moving average of a 1D numpy array.
    :param wavespeeds: Average wave speed data.
    :param window_size: Size of the moving average window.
    :return: Moving average of the input data.
    """
    ret = np.cumsum(wavespeeds, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size