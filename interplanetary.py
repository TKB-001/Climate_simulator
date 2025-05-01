from config import *
from numpy import sin, pi, cos, sqrt, arctan2

"""citation for kepler's equation: N. M. Swerdlow, "Kepler's iterative solution to Kepler's equation," J. Hist. Astron., vol. 31, no. 4, pp. 339-341, 2000,
 doi: 10.1177/002182860003100404"""

def orbital_distance(t, a, e, P_orb):
    """
    t      : time since periastron [seconds]
    a      : semi-major axis [m]
    e      : eccentricity
    P_orb  : orbital period [seconds]
    returns: r(t) in meters
    """
    M = 2*pi * (t % P_orb) / P_orb

    E = M.copy()
    for _ in range(8):
        E = E + (M - (E - e*sin(E))) / (1 - e*cos(E))

    f = 2*arctan2(sqrt(1+e)*sin(E/2),
                     sqrt(1-e)*cos(E/2))

    return a * (1 - e**2) / (1 + e*cos(f))