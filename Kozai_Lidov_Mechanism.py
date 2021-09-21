# Simulation of secular dynamics of a three-body system, 
# using the Hierarchical configuration.

from numpy import loadtxt, float64, str_, degrees, radians
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from kozai.delaunay import TripleDelaunay

# Data has the format:
# [name, a[au], ecc, i[rad], omega[rad], mass[M_Sun]]
Data =  loadtxt("Data.asc", usecols=(0,1,2,3,4,8),
        dtype={'names': ('Name', 'Semi-major axis', 'Eccentricity', 'Inclination', 'Peri', 'Mass'),
               'formats': ("|S15", float64, float64, float64, float64, float64)})

Body_1 = Data[1] # Inner Body
Body_2 = Data[0] # Outer Body

tf = 1e5 # Final time [yr]

# Triple system using the Delaunay orbital elements
System_3Body = TripleDelaunay(a1=Body_1['Semi-major axis'], a2=Body_2['Semi-major axis'],
                        e1=Body_1['Eccentricity'], e2=Body_2['Eccentricity'],
                        g1=degrees(Body_1['Peri']), g2=degrees(Body_2['Peri']),
                        inc=degrees(Body_1['Inclination']-Body_2['Inclination']),
                        m1=1., m2=Body_1['Mass'], m3=Body_2['Mass'])

System_3Body.octupole = False # Not consider the octupole term 

Evolution_Qua = System_3Body.evolve(tf) # Quadrupole Order
print(f'Final time evaluated = {System_3Body.t} yr.')
System_3Body.reset() # Time is set to zero

System_3Body.octupole = True # Turn on the octupole term 
Evolution_Oct = System_3Body.evolve(tf) # Octupole order
print(f'Final time evaluated = {System_3Body.t} yr.')
System_3Body.reset() # Time is set to zero

System_3Body.hexadecapole = True # Turn on the hexadecapole term
Evolution_Hex = System_3Body.evolve(tf) # Hexadecapole order

# Plots
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.formatter.limits']= -3, 3
fig, axs = plt.subplots(2, 2, figsize=(16,8))
# Title, in Name we delete extra characters due to str format.
fig.suptitle('COEs of ' + str_(Body_1['Name'])[2:-1] + ' (Secular)' ,fontsize=20)

# Plot Semi-major axis
axs[0, 0].plot(Evolution_Qua[:, 0],Evolution_Qua[:, 1], color='cyan', label='Quadrupole')
axs[0, 0].plot(Evolution_Oct[:, 0],Evolution_Oct[:, 1], color='darkviolet', label='Octupole')
axs[0, 0].plot(Evolution_Hex[:, 0],Evolution_Hex[:, 1], color='red', label = 'Hexadecapole')
axs[0, 0].set_title('$a$')
axs[0, 0].set_ylabel('[au]')
axs[0, 0].set_xlabel('$t$ [yr]')
axs[0, 0].legend()

# Plot Eccentricity
axs[0, 1].plot(Evolution_Qua[:, 0],Evolution_Qua[:, 2], color='cyan')
axs[0, 1].plot(Evolution_Oct[:, 0],Evolution_Oct[:, 2], color='darkviolet')
axs[0, 1].plot(Evolution_Hex[:, 0],Evolution_Hex[:, 2], color='red')
axs[0, 1].set_title('$e$')
axs[0, 1].set_xlabel('$t$ [yr]')

# Plot Inclination
axs[1, 0].plot(Evolution_Qua[:, 0], radians(Evolution_Qua[:, -1]), color='cyan')
axs[1, 0].plot(Evolution_Oct[:, 0], radians(Evolution_Oct[:, -1]), color='darkviolet')
axs[1, 0].plot(Evolution_Hex[:, 0], radians(Evolution_Hex[:, -1]), color='red')
axs[1, 0].set_title('$\iota$')
axs[1, 0].set_ylabel('[rads]')
axs[1, 0].set_xlabel('$t$ [yr]')

# Plot argument of periapse
axs[1, 1].plot(Evolution_Qua[:, 0],radians(Evolution_Qua[:, 3]), color='cyan')
axs[1, 1].plot(Evolution_Oct[:, 0],radians(Evolution_Oct[:, 3]), color='darkviolet')
axs[1, 1].plot(Evolution_Hex[:, 0],radians(Evolution_Hex[:, 3]), color='red')
axs[1, 1].set_title('$\omega$')
axs[1, 1].set_ylabel('[rads]')
axs[1, 1].set_xlabel('$t$ [yr]')

plt.setp(axs, xlim=[0,tf])
plt.subplots_adjust(wspace=0.4,hspace=0.4)

plt.show()