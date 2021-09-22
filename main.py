from common import *
from conversion import *
from tqdm import tqdm

from numpy import array, zeros, linspace, savetxt, loadtxt, float64, str_, zeros

# Examples of celestial bodies
# Data has format:
# [name, a[au], ecc, i[rad], omega[rad], Omega[rad], Epoch[yr], time[yr], mass[Solar masses], period [yr]]
Data =  loadtxt("Data.asc",
        dtype={'names': ('Name', 'Semi-major axis', 'Eccentricity', 'Inclination', 'Peri', 'Node', 'Epoch', 'Time', 'Mass', 'Period'),
               'formats': ("|S15", float64, float64, float64, float64, float64, float64, float64, float64, float64)})

# Bodies initial data
Body_2 = Data[0] # Body 2
Body_3 = Data[1]  # Body 3
names = ['Sun', str_(Body_2['Name'])[2:-1], str_(Body_3['Name'])[2:-1]] # we delete extra characters due to str format.

k = 10000 # Number of outer periods.
jump = 100 # Jump size to store data in files

# Time's interval of a outer period [yr]
T = max(Body_2['Period'], Body_3['Period'])

# Number of time-iterations in one outer period.
n = 10000

# Discrete time step. 
dt = T/n # [yr]
print(f'dt={dt} and tf={k*T}')
    
# Masses in Solar masses [m1, m2, m3]
masses = array([1., Body_2['Mass'], Body_3['Mass']])

# Initial Conditions
# Transformation of initial conditios from orbital parameters to cartesian coordinates
# Body 2
ini_pos_2, ini_vel_2 = op_to_coords( G*masses[0], Body_2['Semi-major axis'], Body_2['Eccentricity'],
                        Body_2['Inclination'], Body_2['Peri'], Body_2['Node'], Body_2['Epoch'], Body_2['Time'])
# Body 3
ini_pos_3, ini_vel_3 = op_to_coords( G*masses[0], Body_3['Semi-major axis'], Body_3['Eccentricity'],
                        Body_3['Inclination'], Body_3['Peri'], Body_3['Node'], Body_3['Epoch'], Body_3['Time'])

# Array to store the system's evolution
Evolution = zeros([n,3,6])
# initial x, y, z, vx, vy, vz of Body 1
Evolution[0, 0] = array([0., 0., 0., 0., 0., 0.]) # It's in the center and at rest.
# initial x, y, z, vx, vy, vz of Body 2
Evolution[0, 1] = ini_pos_2[0], ini_pos_2[1], ini_pos_2[2],\
                  ini_vel_2[0], ini_vel_2[1], ini_vel_2[2]
# initial x, y, z, vx, vy, vz of Body 3
Evolution[0, 2] = ini_pos_3[0], ini_pos_3[1], ini_pos_3[2],\
                  ini_vel_3[0], ini_vel_3[1], ini_vel_3[2]

# Create video of the first outer period
Vid = int(input('Create a video [1=Yes/0=No]: '))
if Vid:
    # Frequency at which .PNG images are written.
    img_step = 50
    # Folder to save the images
    image_folder = 'images/'
    # Name of the generated video
    video_name = names[0]+'_'+names[1]+'_'+names[2]+'_Initial.mp4'
    # Center of the image
    center = Evolution[0,0,:3] # Central star position.

    Evolution = PEFRL(Acceleration, Evolution[0], masses, n, dt)
    # Generate video
    create_images(Evolution, dt, center, img_step, image_folder, video_name)
    create_video(image_folder, video_name)
else:
    Evolution = PEFRL(Acceleration, Evolution[0], masses, n, dt)

# Create files to store the Evolution of system

fmt='%e', '%d', '%d', '%d' # Format to dt, n, k, jump

# Cartesian Coordinates
FileB0_CC = open('./Files/CC_'+names[0]+'.txt', "w")
FileB1_CC = open('./Files/CC_'+names[1]+'.txt', "w")
FileB2_CC = open('./Files/CC_'+names[2]+'.txt', "w")
FileB0_CC.write('# '+names[0]+'\n' + '# Evolution of cartesian state vectors\n')
FileB1_CC.write('# '+names[1]+'\n' + '# Evolution of cartesian state vectors\n')
FileB2_CC.write('# '+names[2]+'\n' + '# Evolution of cartesian state vectors\n')
savetxt(FileB0_CC,array([[dt, n, k, jump]]), fmt=fmt)
savetxt(FileB1_CC,array([[dt, n, k, jump]]), fmt=fmt)
savetxt(FileB2_CC,array([[dt, n, k, jump]]), fmt=fmt)

#Orbital Parameters
FileB1_OP = open('./Files/OP_'+names[1]+'.txt', "w")
FileB2_OP = open('./Files/OP_'+names[2]+'.txt', "w")
FileB1_OP.write('# '+names[1]+'\n' + '# Evolution of Keplerian Orbit Elements\n')
FileB2_OP.write('# '+names[2]+'\n' + '# Evolution of Keplerian Orbit Elements\n')
savetxt(FileB1_OP,array([[dt, n, k, jump]]), fmt=fmt)
savetxt(FileB2_OP,array([[dt, n, k, jump]]), fmt=fmt) 

# Solution to the problem using PEFRL method
print('\nEvolution:')
Orbital_elements = zeros([n,2,5])
Bar = tqdm(total = k) #Bar changing
for j in range(k):
    savetxt(FileB0_CC, Evolution[::jump,0,:])
    savetxt(FileB1_CC, Evolution[::jump,1,:])
    savetxt(FileB2_CC, Evolution[::jump,2,:])
    for i in range(n):
        Orbital_elements[i,0]=coords_to_op(G*masses[0],Evolution[i,1,:3]-Evolution[i,0,:3] , Evolution[i,1,3:]-Evolution[i,0,3:])
        Orbital_elements[i,1]=coords_to_op(G*masses[0],Evolution[i,2,:3]-Evolution[i,0,:3] , Evolution[i,2,3:]-Evolution[i,0,3:])
    savetxt(FileB1_OP, Orbital_elements[::jump,0])
    savetxt(FileB2_OP, Orbital_elements[::jump,1]) 
    Bar.update(1)
    Evolution[0] = Evolution[n-1]
    Evolution = PEFRL(Acceleration, Evolution[0], masses, n, dt)
Bar.close()
FileB0_CC.close()
FileB1_CC.close()
FileB2_CC.close()
FileB1_OP.close()
FileB2_OP.close()

# Final Video
if Vid:
    # Folder to save the images
    image_folder = 'images/'
    # Name of the generated video
    video_name = names[0]+'_'+names[1]+'_'+names[2]+'_Final.mp4'
    # Center of the image
    center = Evolution[0,0,:3] # Central star position.
    #Generate video
    create_images(Evolution, dt, center, img_step, image_folder, video_name)
    create_video(image_folder, video_name)
print('\nEnd.')