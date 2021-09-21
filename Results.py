import matplotlib.pyplot as plt
plt.style.use('dark_background')
from common import Constants
from Plots import Plot_OrbElem
from tqdm import tqdm

from numpy import loadtxt, linspace, array, zeros

# Body's name
Names = ['Sun', 'Jupiter','Kozai'] # Remember to update the 3rd body!!!
#Body's masses [M_sun]
Masses = [1., 9.54792e-04, 1.0e-16]

# Path to the files
path_OP_B1 = './Files/OP_' + Names[1] + '.txt'
path_OP_B2 = './Files/OP_' + Names[2] + '.txt'
path_CC_B0 = './Files/CC_' + Names[0] + '.txt'
path_CC_B1 = './Files/CC_' + Names[1] + '.txt'
path_CC_B2 = './Files/CC_' + Names[2] + '.txt'

# dt: Discrete time step.
# n: Number of time-iterations in one outer period.
# k: Number of outer periods.
# jump: Jump size to store data in files
dt, n, k, jump =  loadtxt(path_OP_B1, max_rows=1, unpack=True,
                          dtype={'names': ('dt','n','k','jump'), 'formats': (float, int, int, int)})

# Array to store time information
it = int(n/ jump)
t = linspace(0, dt*n*k, k*it)

# Plots of orbital elements.
Plot_OrbElem(t, loadtxt(path_OP_B1, skiprows=3), Names[1])
Plot_OrbElem(t, loadtxt(path_OP_B2, skiprows=3), Names[2])

# Energy and Angular momentum of the system

# Plots
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.formatter.limits']= -3, 3
fig, axs = plt.subplots(1, 2, figsize=(16,8))

axs[0].set_title('Energy')
axs[0].set_ylabel('$M_\odot au^2 yr ^ {-2}$')
axs[0].set_xlabel('$t$ [yr]')

axs[1].set_title('Angular momentum')
axs[1].set_ylabel('$M_\odot au^2 yr ^ {-1}$')
axs[1].set_xlabel('$t$ [yr]')

plt.setp(axs, xlim=[0,t[-1]])

Energy = zeros(it) # Energy 
AngMom = zeros(it) # Angular momentum
print('Calculating Energy and Angular Momentum...')
Bar = tqdm(total = k) #Bar changing
for i  in range(k):
    Evolution_B0 = loadtxt(path_CC_B0, skiprows= 3+i*it, max_rows=it)
    Evolution_B1 = loadtxt(path_CC_B1, skiprows= 3+i*it, max_rows=it)
    Evolution_B2 = loadtxt(path_CC_B2, skiprows= 3+i*it, max_rows=it)
    for j in range(it):
        Energy[j], AngMom[j] = Constants(array([Evolution_B0[j],Evolution_B1[j], Evolution_B2[j]]), Masses)
    axs[0].plot(t[i*it:(i+1)*it],Energy, color='cyan')
    axs[1].plot(t[i*it:(i+1)*it],AngMom, color='cyan')
    Bar.update(1)
Bar.close()
plt.show()