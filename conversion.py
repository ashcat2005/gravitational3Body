# Conversion between orbital parameters and cartesian positions 
# and cartesian velocities.

from numpy.linalg import norm
from numpy import array, dot, cross
from math import sin, cos, acos, pi, sqrt
from scipy.optimize import newton

def op_to_coords(mu, a, ecc, i, omega, Omega, t_0, t):
    '''
    ------------------------------------------------------------------
    op_to_coords(mu, a, ecc, i, omega, Omega, t_0, t)
    Transforms from the orbital parameters to the coordinates of the 
    position vector and the components of the velocity at time t.
    Verify the system of units for the arguments mu, a and t.
    ------------------------------------------------------------------
    Arguments: 
        a:      Semi-major axis
        ecc:    Eccentricity
        i:      Inclination
        omega:  Argument of the pericenter
        Omega:  Longitude of the ascending node
        t_0:    Epoch
        t:      Time to calculate the position and velocity
    ------------------------------------------------------------------
    Returns: 
        position:  Numpy array with the components of the position in 
                   cartesian coordinates [x,y,z].
        Velocity:  Numpy array with the components of the velocity in 
                   cartesian coordinates [v_x,v_y,v_z].
    ------------------------------------------------------------------
    '''
    # Mean Motion
    n = sqrt(mu/a**3)
    # Mean Anomaly
    M = n*(t-t_0)
    # Eccentric Anomaly
    # Kepler problem
    f = lambda x: x - ecc*sin(x) - M
    f_prime = lambda x: 1. - ecc*cos(x)
    f_2prime = lambda x: ecc*sin(x)
    E = newton(f, 0., fprime=f_prime, fprime2=f_2prime) #Halley’s method
    # Radial coordinate at time t
    r = a*(1-ecc*cos(E))
    # Coordinates of the body at time t in the orbital plane
    r_XYZ = array([a*(cos(E) - ecc), a*sqrt(1 - ecc**2)*sin(E), 0.])
    # Components of the velocity of the body at time t in the orbital plane
    v_XYZ = array([-(sqrt(mu*a)/r)*sin(E),
                   (sqrt(mu*a*(1 - ecc**2))/r)*cos(E),
                   0.])
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

def coords_to_op(mu, r_xyz, v_xyz):
    '''
    ----------------------------------------------------------------------------
    coords_to_op(mu, r_xyz, v_xyz)
    Transforms from the coordinates of the position and the components of the
    velocity at t=0. to the orbital parameters.
    Verify the system of units for the arguments.
    ----------------------------------------------------------------------------
    Arguments: 
        position:  Numpy array with the components of the position in 
                   cartesian coordinates [x,y,z].
        Velocity:  Numpy array with the components of the velocity in 
                   cartesian coordinates [v_x,v_y,v_z].        
    ----------------------------------------------------------------------------
    Returns: 
        a:      Semi-major axis
        ecc:    Eccentricity
        i:      Inclination
        omega:  Argument of the pericenter
        Omega:  Longitude of the ascending node
    ----------------------------------------------------------------------------
    '''
    # Norm of the position and velocity vectors
    r = norm(r_xyz)
    v = norm(v_xyz)
    # Radial velocity
    v_r = dot(r_xyz,v_xyz)/r
    # Angular momentum
    h_xyz = cross(r_xyz, v_xyz)
    # Norm of the angular momentum
    h = norm(h_xyz)
    # Inclination of the orbit
    i = acos(h_xyz[2]/h)
    # Line of the Nodes
    N = sqrt(h_xyz[1]**2+h_xyz[0]**2)
    # Longitude of the ascending node
    if h_xyz[0] >= 0:
        Omega = acos(-h_xyz[1]/N)
    else:
        Omega = 2*pi - acos(-h_xyz[1]/N)
    
    # Eccentricity vector
    ecc_xyz = (1/mu)*((v**2 - mu/r)*r_xyz - r*v_r*v_xyz)
    # Eccentricity scalar
    ecc = norm(ecc_xyz)
    
    # Semi-major axis
    a = h**2/(mu*(1-ecc**2))
    
    # Argument of the pericenter
    aux =  (-h_xyz[1]*ecc_xyz[0]+h_xyz[0]*ecc_xyz[1])/(N*ecc)
    if abs(aux) >1.:
        omega = 0.
    else:
        if ecc_xyz[2] >=0.:
            omega = acos(aux)
        else:
            omega = 2*pi - acos(aux)
    
    return a, ecc, i, omega, Omega

if __name__ == "__main__":
    '''
    Example: test comet
    '''
    # Working with the system of units au, yr, and solar masses
    G = 4*pi**2 # Newtonian Gravitational Constant
    sunM = 1.   # Mass of the Sun
    mu = G*sunM
    
    # Orbital Parameters
    a = 0.4667      # semi-major axis [au]
    ecc = 0.04      # eccentricity
    i = 74*pi/180.          # orbital plane inclination [rad]
    Omega = 58.42*pi/180.   # Argument of the ascendent node [rad]
    omega = 111.33*pi/180   # Argument of the pericenter [rad]
    t_0 = 0.    # Epoch [yr]
    t =0.       # Time to calculate the position and velocity [yr]

    r_xyz, v_xyz = op_to_coords(mu, a, ecc, i, omega, Omega, t_0, t)

    print(f'\nr = {r_xyz}')
    print(f'\nv = {v_xyz}\n')

    new_a, new_ecc, new_i, new_omega, new_Omega = coords_to_op(mu, r_xyz, v_xyz)

    print(f'a = {a}  -  {new_a}')
    print(f'e = {ecc}  -  {new_ecc}')
    print(f'i = {i}  -  {new_i}')
    print(f'w = {omega}  -  {new_omega}')
    print(f'Ω = {Omega}  -  {new_Omega}')