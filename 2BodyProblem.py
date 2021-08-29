#Simulation of two bodies in gravitational interaction.

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from conversion import *
from Plots import *

# Gravitational constant in units of [au^3 M_sun^-1 yr-2]
G = 4*np.pi**2

#Bodies data as [x[au], y[au], z[au], vx[au/yr], vy[au/yr], vz[au/yr], mass[solar masses], orbital period[yr]]
Sun = np.array([0.,0.,0.,0.,0.,0.,1.,0.])
Mercury = np.array([-2.503321047836E-01, 1.873217481656E-01, 1.260230112145E-01,-8.90756486, -6.75780661, -2.68592451, 1.66014e-07, 0.24094])
Venus = np.array([1.747780055994E-02, -6.624210296743E-01, -2.991203277122E-01, 7.3360674, 0.30554196, -0.32681492, 2.08106e-06, 0.61603, ])
Earth_Moon = np.array([-9.091916173950E-01, 3.592925969244E-01, 1.557729610506E-01,-2.5880511,-5.31659521,-2.30501358, 3.00273e-6, 1.])
Mars = np.array([1.203018828754E+00,7.270712989688E-01, 3.009561427569E-01, -2.60215337,4.25985033,2.02420998, 3.23237e-07, 1.8809])
Jupiter = np.array([3.733076999471E+00, 3.052424824299E+00, 1.217426663570E+00, -1.85782081, 2.00651219, 0.90532114, 9.54791e-04, 11.8618])
Saturn = np.array([6.164433062913E+00, 6.366775402981E+00, 2.364531109847E+00, -1.61686412, 1.23965502, 0.58156154, 2.85885e-04, 29.4571])
Uranus = np.array([1.457964661868E+01, -1.236891078519E+01, -5.623617280033E+00, 0.96698158, 0.90852515, 0.3842352, 4.36624e-05, 84.0182])
Neptune = np.array([1.695491139909E+01, -2.288713988623E+01, -9.789921035251E+00, 0.9381808, 0.61427667, 0.22811637, 5.15138e-05, 164.7946])
Pluto = np.array([-9.707098450131E+00,-2.804098175319E+01,-5.823808919246E+00, 1.108187, -0.4059004, -0.46087813, 6.58086e-09, 248.])
Halley = np.array([0.33126099, -0.45385512, 0.16628889,-9.01382671, -7.04649889, -1.27585467,1.106378E-16,75.32])
Geographos = np.array([-0.21765384, -0.77581474, -0.18955119, 7.66669988, -2.20533078, 0.22284989, 1.0E-17,1.39])
Hidalgo = np.array([0.56084979, 1.5044718, 1.09763743,-5.37114217, 0.42400368, 2.16328483, 1.0E-17,13.72])
a_2021PH27 = np.array([0.09174773, 0.09731134, 0.01034353, -14.54945423, 12.47980066, 11.64527326, 1.0E-17,0.31])


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
        q0[1] = Body 2 where
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
    a = np.zeros([2, 3])
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
    speed2 = np.array(np.sum(q[:,3:]**2, axis=1))
    r= LA.norm(q[0,:3] - q[1,:3])
    U = -G*mass[1]*mass[0]/r
    E = np.sum(0.5*speed2*mass) + 2*U # Total energy
    L = np.cross(q[0,:3],(mass[0]*q[0,3:]))+ np.cross(q[1,:3],(mass[1]*q[1,3:])) #Total angular momentum vector
    
    return E, LA.norm(L)

def Runge_Lenz(q, mass):
    '''
    ----------------------------------------------------
    Runge_Lenz(q, mass)
    Calculates the magnitude of the Runge-Lenz vector 
    of 2 particles interacting gravitationally.
    ----------------------------------------------------
    Arguments:
    q : Numpy array with the position and velocity of 
        the particles with the format
        q = [[x1, y1, z1, vx1, vy1, vz1],
             [x2, y2, z2, vx2, vy2, vz2]]
    mass : NumPy array with the masses of the particles.
        mass = [m1, m2]
    ----------------------------------------------------
    Returns:
    A = Magnitude of the Runge-Lenz vector.
    ----------------------------------------------------
    '''
    L1 = np.cross(q[0,:3],(mass[0]*q[0,3:])) # r1 x p1
    L2 = np.cross(q[1,:3],(mass[1]*q[1,3:])) # r2 x p2
    r1 =  LA.norm(q[0,:3])
    r2 =  LA.norm(q[1,:3])
    
    if (r1!=0 and r2!=0):
        A1 = np.cross(mass[0]*q[0,:3], L1) - mass[0]*G*q[0,:3]/r1
        A2 = np.cross(mass[1]*q[1,:3], L2) - mass[1]*G*q[1,:3]/r2
    elif r1==0 and r2!= 0:
        A1 = np.cross(mass[0]*q[0,:3], L1) 
        A2 = np.cross(mass[1]*q[1,:3], L2) - mass[1]*G*q[1,:3]/r2
    elif r1!=0 and r2==0:
        A1 = np.cross(mass[0]*q[0,:3], L1) - mass[0]*G*q[0,:3]/r1
        A2 = np.cross(mass[1]*q[1,:3], L2)
    else:
        A1 = np.cross(mass[0]*q[0,:3], L1)
        A2 = np.cross(mass[1]*q[1,:3], L2)
    
    return LA.norm(A1+A2)

def PEFRL(ODE, q0, mass, n, t):
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
    t:  array (t0,tf) whit the initial and final time.
    ----------------------------------------------------
    Returns:
    q = NumPy array with system's evolution since t0 to tf
    ----------------------------------------------------
    '''
    # Arrays to store the solution
    q = np.zeros([n, 2, 6])  
    q[0] = q0
    
    #stepsize for the iteration
    h = (t[1]-t[0])/n

    #Parameter
    xi = 0.1786178958448091E+00
    gamma = -0.2123418310626054E+00
    chi = -0.6626458266981849E-01 
    
    # Main Loop
    for i in range(n-1):
        x_1 = q[i,:,:3] + xi*h*q[i,:,3:]
        F = ODE(x_1, mass)
        v_1 = q[i,:,3:] + 0.5*(1.-2*gamma)*h*F
        x_2 = x_1 + chi*h*v_1
        F = ODE(x_2, mass)
        v_2 = v_1 + gamma*h*F
        x_3 = x_2 + (1.-2*(chi+xi))*h*v_2
        F = ODE(x_3, mass)
        v_3 = v_2 + gamma*h*F
        x_4 = x_3 + chi*h*v_3
        F = ODE(x_4, mass)
        q[i+1,:,3:] = v_3 + 0.5*(1.-2*gamma)*h*F
        q[i+1,:,:3] = x_4 + xi*h*q[i+1,:,3:]

    return q

# Bodies initial data
Body_1 = Sun # Body 1
Body_2 = Jupiter # Body 2
names = ['Sun', 'Jupiter']

# Time's interval [yr]
t = np.array([0.,max(Body_1[-1],Body_2[-1])])

# Number of time-iterations executed by the program.
n = 10000 # Time steps

print(f'\ntf={t[1]} [yr] and dt={(t[1]-t[0])/n} [yr]\n')

# Array to store time information
t1 = np.linspace(t[0], t[1], n) 

# Masses in Solar masses
m1 = Body_1[6]
m2 = Body_2[6]
masses = np.array([m1, m2])

# Initial Conditions
Evolution = np.zeros([n,2,6]) 
Evolution[0, 0] = Body_1[:6] # initial x, y, z, vx, vy to Body 1
Evolution[0, 1] = Body_2[:6] # initial x, y, z, vx, vy to Body 2

# Solution to the problem using PEFRL method
print('Evolution in process...')
Evolution = PEFRL(Acceleration, Evolution[0], masses, n, t)
print('The process has finished.')

#Plot Orbit
Plot_orbit(Evolution, names)

# Array of Energy and Angular momentum of the system
Energy = np.zeros(n) #Energy 
AngMom = np.zeros(n) #Angular momentum
print('Calculating Energy and Angular Momentum...')
for i in range(n):
    Energy[i], AngMom[i] = Constants(Evolution[i], masses)
print('Energy and Angular momentum calculated.')

#Plot Energy
fig, ax = plt.subplots( figsize=(10,7))

ax.plot(t1, Energy, color='crimson')
ax.set_title('Energy for system: '+ names[0] +'-'+names[1])
ax.set_xlabel(r'$t~[yr]$')
ax.set_ylabel(r'$E$')
plt.show()

#Plot Angular momentum
fig, ax = plt.subplots( figsize=(10,7))

ax.plot(t1, AngMom, color='crimson')
ax.set_title('Angular Momentum for system: '+ names[0] +'-'+names[1])
ax.ticklabel_format(useMathText=True)
ax.set_xlabel(r'$t~[yr]$')
ax.set_ylabel(r'$L$')
plt.show()

#Orbital Elements
orbital_elements = np.zeros([n,5]) #  Evolution of orbital elements
mu = G*m1
print('Transforming to orbital elements...')
for i in range(n):
    orbital_elements[i]=coords_to_op(mu,Evolution[i,1,:3]-Evolution[i,0,:3],Evolution[i,1,3:]-Evolution[i,0,3:])
print('Ok!')

#Plot Orbital elements
Plot_OrbElem(t1,orbital_elements)