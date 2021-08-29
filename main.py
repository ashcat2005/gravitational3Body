from common import *
from conversion import *
from Plots import *

from numpy import array, zeros, linspace

#Bodies data as [x, y, z, vx, vy, vz, mass, orbital period, Apsis]

Sun = array([0.,0.,0.,0.,0.,0.,1.,0., 0.])
Jupiter = array([-2.11654703, 4.47630812, 0.0287253, -2.61291102,-1.23587974, 0.06368842, 1.0e-01, 12., 5.46])
Hidalgo = array([0.56084979, 1.5044718, 1.09763743,-5.37114217, 0.42400368, 2.16328483, 1.0E-17,13.72, 9.519253976700893])
a_2021PH27 = array([0.09174773, 0.09731134, 0.01034353, -14.54945423, 12.47980066, 11.64527326, 1.0e-5,0.31, .7886707956777496]) #Real approximate mass 1.0e-17
# Bodies initial data
Body_1 = Sun # Body 1
Body_2 = Jupiter # Body 2
Body_3 = a_2021PH27  # Body 3
names = ['Sun', 'Jupiter', '2021PH27']

# Time's interval [yr]
t = array([0.,max(Body_1[7],Body_2[7], Body_3[7])])

# Number of time-iterations executed by the program.
n = 100000 # Time steps

# Discrete time step. 
dt = (t[1]-t[0])/n # [yr]
print(f'dt={dt} and tf={t[1]}')

# Array to store time information
t1 = linspace(t[0], t[1], n)
    
# Masses in Solar masses
m1 = Body_1[6]
m2 = Body_2[6]
m3 = Body_3[6]
masses = array([m1, m2, m3])

#Array of Energy and Total angular momentum of the system
Consts= zeros([n+1,2])

# Initial Conditions
Evolution = zeros([n,3,6]) #  Motion information FR
Evolution[0, 0] = Body_1[:6] # initial x, y, z, vx, vy to particle 1
Evolution[0, 1] = Body_2[:6] # initial x, y, z, vx, vy to particle 2
Evolution[0, 2] = Body_3[:6] # initial x, y, z, vx, vy to particle 3


# Solution to the problem using LeapFrog method
print('Evolution in process...')
Evolution = PEFRL(Acceleration, Evolution[0], masses, n, t)
print('The process has finished.')

Plot_orbit(Evolution, names)
# Frequency at which .PNG images are written.
img_step = 50
# Folder to save the images
image_folder = 'images/'
# Name of the generated video
video_name = names[0]+'-'+names[1]+'-'+names[2]+'.mp4'
# Center of the image
center = array([0., 0., 0.])

#Generate video
#create_images(q, dt, center, img_step, image_folder, video_name)
#create_video(image_folder, video_name)

#Orbital parameters
print('Transforming to orbital elements...')
Orbital_elements = zeros([n,2,5])
for i in range(n):
    Orbital_elements[i,0]=coords_to_op(G*m1,Evolution[i,1,:3]-Evolution[i,0,:3] , Evolution[i,1,3:]-Evolution[i,0,3:])
    Orbital_elements[i,1]=coords_to_op(G*m1,Evolution[i,2,:3]-Evolution[i,0,:3] , Evolution[i,2,3:]-Evolution[i,0,3:])
print('Ok!')

Plot_OrbElem(t1,Orbital_elements[:,1])

