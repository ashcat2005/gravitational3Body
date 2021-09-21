# Simulation of two bodies interacting gravitationally.

from numpy import pi, loadtxt, float64, zeros, array, sum, cross, str_, linspace
from numpy import linalg as LA
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from conversion import *
from Plots import *

# Gravitational constant in units of [au^3 M_sun^-1 yr-2]
G = 4*pi**2

# Examples of celestial bodies.
# Data has the format:
# [name, a[au], ecc, i[rad], omega[rad], Omega[rad], Epoch[yr], time[yr], mass[M_Sun], period [yr]]
Data =  loadtxt("Data.asc",
        dtype={'names': ('Name', 'Semi-major axis', 'Eccentricity', 'Inclination', 'Peri', 'Node', 'Epoch', 'Time', 'Mass', 'Period'),
               'formats': ("|S15", float64, float64, float64, float64, float64, float64, float64, float64, float64)})

def Acceleration(q0, mass):
    '''
    ----------------------------------------------------
    Acceleration(q0, mass):
    Calculates the acceleration due to gravitation for
    each body in the 2 body system.
    ----------------------------------------------------
    Arguments:
    q0: Numpy array with position data:
        q0[0] = Body 1
        q0[1] = Body 2, where
        q0[i] = [x_i, y_i, z_i] for i=[0,1]
    mass: masses of the bodies
        mass = [m1, m2]
    ----------------------------------------------------
    Returns:
    a = NumPy array with the components of the 
        acceleration
        a[i] = [ax_i, ay_i, az_i] for i=[0,1]
    ----------------------------------------------------
    '''
    a = zeros([2, 3])
    Deltaxyz = q0[0] - q0[1]
    r = LA.norm(Deltaxyz)  # Distance between particles
    a[0, :] = -G * Deltaxyz * mass[1]/r**3  # Acceleration Body_1
    a[1, :] = G * Deltaxyz * mass[0]/r**3  # Acceleration Body_2

    return a

def Constants(q, mass):
    '''
    ----------------------------------------------------
    Constants(q, mass)
    Calculates the total energy and total angular 
    momentum of 2 particles interacting gravitationally.
    ----------------------------------------------------
    Arguments:
    q:  Numpy array with the position and velocity of each
        particle following the next format:
        q = [[x1, y1, z1, vx1, vy1, vz1],
             [x2, y2, z2, vx2, vy2, vz2]]
    mass: NumPy array with the masses of the particles.
        mass = [m1, m2]
    ----------------------------------------------------
    Returns:
    E = Total energy of the system.
    L = Magnitude of the total angular momentum of the 
        system.
    ----------------------------------------------------
    '''
    speed2 = array(sum(q[:,3:]**2, axis=1))
    U = -G*mass[1]*mass[0]/LA.norm(q[0,:3] - q[1,:3])
    E = sum(0.5*speed2*mass) + 2*U # Total energy
    L = cross(q[0,:3],(mass[0]*q[0,3:]))+ cross(q[1,:3],(mass[1]*q[1,3:])) # Total angular momentum vector
    
    return E, LA.norm(L)

def PEFRL(ODE, q0, mass, n, dt):
    '''
    ----------------------------------------------------
    PEFRL(ODE, q0, mass, n, t)
    Position Extended Forest-Ruth Like (PEFRL) method
    for time evolution.
    ----------------------------------------------------
    Arguments:
    ODE:function defining the system of ODEs
    q0: numpy array with the initial values of
        the functions in the ODEs system
        q[0] : [x1, y1, z1, vx1, vy1, vz1]
        q[1] : [x2, y2, z2, vx1, vy2, vz2]
    mass: masses of the particles
        mass = [m1, m2]
    n:  Number of steps in the grid
    dt:  Stepsize for the iteration
    ----------------------------------------------------
    Returns:
    q = NumPy array with system's evolution since t0 to tf
    ----------------------------------------------------
    '''
    # Arrays to store the solution
    q = zeros([n, 2, 6])  
    q[0] = q0

    # Parameters
    xi = 0.1786178958448091E+00
    gamma = -0.2123418310626054E+00
    chi = -0.6626458266981849E-01 
    
    # Main Loop
    for i in range(n-1):
        x_1 = q[i,:,:3] + xi*dt*q[i,:,3:]
        F = ODE(x_1, mass)
        v_1 = q[i,:,3:] + 0.5*(1.-2*gamma)*dt*F
        x_2 = x_1 + chi*dt*v_1
        F = ODE(x_2, mass)
        v_2 = v_1 + gamma*dt*F
        x_3 = x_2 + (1.-2*(chi+xi))*dt*v_2
        F = ODE(x_3, mass)
        v_3 = v_2 + gamma*dt*F
        x_4 = x_3 + chi*dt*v_3
        F = ODE(x_4, mass)
        q[i+1,:,3:] = v_3 + 0.5*(1.-2*gamma)*dt*F
        q[i+1,:,:3] = x_4 + xi*dt*q[i+1,:,3:]

    return q

# Data of bodies
Body_2 = Data[0] # Body 2
names = ['Sun', str_(Body_2['Name'])[2:-1]] # Here, we dalete a extra characters
                                               # of the 2nd body's name

# Number of time-iterations.
n = 10000

# Array to store time information [yr]
t = linspace(0., 1e1*Body_2['Period'], n)

dt = (t[-1]-t[0])/n #Stepsize for the iteration

print(f"\ntf={t[-1]} yr and dt={dt} yr\n") 

# Masses [M_sun]
masses = array([1., Body_2['Mass']])

# Initial Conditions
# Transformation the initial conditios from orbital parameters to cartesian coordinates
ini_pos_2, ini_vel_2 = op_to_coords( G*masses[0], Body_2['Semi-major axis'], Body_2['Eccentricity'],
                        Body_2['Inclination'], Body_2['Peri'], Body_2['Node'], Body_2['Epoch'], Body_2['Time'])

# Array to store the system's evolution
Evolution = zeros([n,2,6])
# initial x, y, z, vx, vy, vz of Body 1
Evolution[0, 0] = array([0., 0., 0., 0., 0., 0.]) # It's in the center and at rest
# initial x, y, z, vx, vy, vz of Body 2
Evolution[0, 1] = ini_pos_2[0], ini_pos_2[1], ini_pos_2[2],\
                  ini_vel_2[0], ini_vel_2[1], ini_vel_2[2]

# Solution to the problem using PEFRL method
print('Evolution in process...')
Evolution = PEFRL(Acceleration, Evolution[0], masses, n, dt)
print('The process has finished.')

# Plot Orbit
Plot_orbit(Evolution, names)

# Array of Energy and Angular momentum of the system
Energy = zeros(n) # Energy 
AngMom = zeros(n) # Angular momentum
print('Calculating Energy and Angular Momentum...')
for i in range(n):
    Energy[i], AngMom[i] = Constants(Evolution[i], masses)
print('Energy and Angular momentum calculated.')

# Energy and Angular momentum of the system

# Plots
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.formatter.limits']= -3, 3
fig, axs = plt.subplots(1, 2, figsize=(16,8))

axs[0].plot(t,Energy, color='cyan')
axs[0].set_title('Energy')
axs[0].set_ylabel('$M_\odot au^2 yr ^ {-2}$')
axs[0].set_xlabel('$t$ [yr]')

axs[1].plot(t,AngMom, color='cyan')
axs[1].set_title('Angular momentum')
axs[1].set_ylabel('$M_\odot au^2 yr ^ {-1}$')
axs[1].set_xlabel('$t$ [yr]')

plt.setp(axs, xlim=[0,t[-1]])
plt.show()

# Orbital Elements
Orbital_elements = zeros([n,5]) # Evolution of orbital elements
print('Transforming to orbital elements...')
for i in range(n):
    Orbital_elements[i]=coords_to_op(G*masses[0],Evolution[i,1,:3]-Evolution[i,0,:3],Evolution[i,1,3:]-Evolution[i,0,3:])
print('Ok!')

# Plot Orbital elements
Plot_OrbElem(t,Orbital_elements, names[1])