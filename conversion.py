# Conversion between orbital parameters and initial conditions

from numpy import pi, sqrt, sin, cos, array
from scipy.optimize import newton


################################################################################
# Global Constants
################################################################################

# Working with the system of units au, yr, solar masses

G = 4*pi**2 #6.6725*10**-11   # Newtonian Gravitational Constant
c = 63197.8       # Speed of light in au/yr units
SunM = 1. # Mass of the Sun

################################################################################

def transformation(a, ecc, i, omega, Omega, t_0, t):
    # Mean Motion
    n = sqrt(G*SunM/a**3)
    # Mean Anomaly
    M = n*(t-t_0)
    # Eccentric Anomaly
    # Kepler problem
    f = lambda x: x - ecc*sin(x) - M
    f_prime = lambda x: 1. - ecc*cos(x)
    E = newton(f, 0., f_prime)
    # Radial coordinate at time t
    r = a*(1-ecc*cos(E))
    # Coordinates of the body at time t in the orbital plane
    r_XYZ = array([[a*(cos(E) - ecc)],[a*sqrt(1 - ecc**2)*sin(E)], [0.]])
    # Components of the velocity of the body at time t in the orbital plane
    v_XYZ = array([[(sqrt(G*SunM*a)/r)*sin(E)],
                   [(sqrt(G*SunM*a*(1 - ecc**2))/r)*cos(E)],
                   [0.]])
    # Rotation Matrix to transform the components of a vector from the
    # orbital plane to the general reference system
    R = array([[cos(omega)*cos(Omega) - sin(omega)*sin(Omega)*cos(i),
                -sin(omega)*cos(Omega) - cos(omega)*sin(Omega)*cos(i),
                sin(Omega)*sin(i)],
               [cos(omega)*sin(Omega) + sin(omega)*cos(Omega)*cos(i),
                -sin(omega)*sin(Omega) + cos(omega)*cos(Omega)*cos(i),
                -cos(Omega)*sin(i)],
               [sin(omega)*sin(i),
                cos(omega)*sin(i),
                cos(i)]])
    
    return R.dot(r_XYZ), R.dot(v_XYZ)


if __name__ == "__main__":
    # Orbital Parameters
    a = 1.         # semi-major axis [au]
    ecc = 0.0167   # eccentricity
    i = pi/4.      # orbital plane inclination
    Omega = 0.     # Argument of the ascendent node
    omega = 0.     # Argument of the pericenter
    t_0 = 0.       # Epoch [yr]
    t = 0.25       # Time to calculate the position and velocity [yr]
    
    r_xyz, v_xyz = transformation(a, ecc, i, omega, Omega, t_0, t)
    print(r_xyz)
    print('')
    print(v_xyz)


