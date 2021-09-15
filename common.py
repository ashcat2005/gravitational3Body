from numpy import pi, array, zeros, sum, cross, amin, amax, linspace
from numpy import linalg as LA
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

# Gravitational constant in units of [au^3 M_sun^-1 yr-2]
G = 4*pi**2

#################################################---EXAMPLES---#################################################
#Some examples of celestial bodies = [x, y, z, vx, vy, vz, mass, orbital period]
Sun = array([0.,0.,0.,0.,0.,0.,1.,0.])
Jupiter = array([3.733076999471E+00, 3.052424824299E+00, 1.217426663570E+00, -1.85782081, 2.00651219, 0.90532114, 9.54791e-04, 11.8618])
Halley = array([0.33126099, -0.45385512, 0.16628889,-9.01382671, -7.04649889, -1.27585467,1.106378E-16,75.32])
Geographos = array([-0.21765384, -0.77581474, -0.18955119, 7.66669988, -2.20533078, 0.22284989, 1.0E-17,1.39])
a_2021PH27 = array([0.09174773, 0.09731134, 0.01034353, -14.54945423, 12.47980066, 11.64527326, 1.0e-5,0.31, .7886707956777496]) #Real approximate mass 1.0e-17

def Acceleration(q0, mass):
    '''
    --------------------------------------------------
    Acceleration(q0, mass):
    Calculates the acceleration due to gravitation for 
    each body in the 3 body system.
    --------------------------------------------------
    Arguments:
    q0: numpy array with the initial condition data:
        q0[0] = particle 1
        q0[1] = particle 2
        q0[2] = particle 3, where:
        q0[i] = [x_i, y_i, z_i] for i=0,1,2
    mass: masses of the particles
        mass = [m1, m2, m3]
    --------------------------------------------------
    Returns:
    a = NumPy array with the components of the 
        acceleration
        a[i] = [ax_i, ay_i, az_i] for i=0,1,2
    '''
    a = zeros([3, 3])
    Deltaxyz = q0[0] - q0[1], q0[0] - q0[2], q0[1] - q0[2]    
    r = LA.norm(Deltaxyz[0]), LA.norm(Deltaxyz[1]), LA.norm(Deltaxyz[2]) # Distance between particles
    a[0,:] = -G * Deltaxyz[0] * mass[1] / r[0]**3 - G * Deltaxyz[1] * mass[2]/r[1]**3
    a[1,:] = G * Deltaxyz[0] * mass[0]/r[0]**3 - G * Deltaxyz[2] * mass[2]/r[2]**3
    a[2,:] = G * Deltaxyz[1] * mass[0]/r[1]**3 + G * Deltaxyz[2] * mass[1]/r[2]**3  
    return a

def Constants(q, mass):
    '''
    --------------------------------------------------
    Constants(q, mass)
    Calculates the total energy and total angular 
    momentum of 3 particles interacting gravitationally.
    --------------------------------------------------
    Arguments:
    q: Numpy array with the position and velocity of 
        the particles with the format
        q = [[x1, y1, z1, vx1, vy1, vz1],
             [x2, y2, z2, vx2, vy2, vz2],
             [x3, y3, z3, vx3, vy3, vz3]]
    mass: NumPy array with the masses of the particles.
        m = [m1, m2, m3]
    --------------------------------------------------
    Returns:
    E = Total energy of the system
    L = Magnitude of the total angular momentum the 
        system
    --------------------------------------------------
    '''
    speed2 = array(sum(q[:,3:]**2, axis=1))
    r = LA.norm(q[0,:3] - q[1,:3]), LA.norm(q[0,:3] - q[2,:3]), LA.norm(q[1,:3] - q[2,:3])
    U = array([mass[0]*mass[1]/r[0], mass[0]*mass[2]/r[1], mass[1]*mass[2]/r [2]])
    E = sum(0.5*speed2*mass-2*G*U) # Total energy
    L = cross(q[0,:3],(mass[0]*q[0,3:]))+ cross(q[1,:3],(mass[1]*q[1,3:]))\
        + cross(q[2,:3],(mass[2]*q[2,3:]))#Total angular momentum vector
    
    return E, LA.norm(L)

def PEFRL(ODE, q0, mass, n, t):
    '''
    --------------------------------------------------
    PEFRL(ODE, q0, mass, n, t)
    --------------------------------------------------
    Position Extended Forest-Ruth Like (PEFRL) method 
    for time evolution.
    Arguments:
    ODE: function defining the system of ODEs
    q0: numpy array with the initial values of
        the functions in the ODEs system
        q[0] : [x1, y1, z1, vx1, vy1, vz1]
        q[1] : [x2, y2, z2, vx1, vy2, vz2]
    mass: masses of the particles
        mass = [m1, m2]
    n:  Number of steps in the grid
    t: array (t0,tf) whit the initial and 
       final time.
    --------------------------------------------------
    Returns:
    q = NumPy array with the components of the 
        solution since t0 to tf
    '''
    # Arrays to store the solution
    q = zeros([n, 3, 6])
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
            
def create_images(evolution, dt, center, img_step, image_folder='images/', video_name='my_video.mp4'):
    '''
    This function evolves the system in time using the PEFRL algorithm
    '''
    
    # Limits for the axes in the plot
    boundary = 1.1*max(abs(amin(evolution[:,:,:3])),abs(amax(evolution[:,:,:3])))
    lim_inf = [center[0]-boundary, center[1]-boundary, center[2]-boundary]
    lim_sup = [center[0]+boundary, center[1]+boundary, center[2]+boundary]
    
    #Write the image files
    for i in range(img_step,len(evolution),img_step):       
        print("Writing image at iteration {0}".format(i))
        plot_bodies(evolution[:i], i//img_step, 2*i*dt, lim_inf, lim_sup, image_folder)

def plot_bodies(bodies, i, time, lim_inf, lim_sup, image_folder='images/'):
    '''
    Writes an image file with the current position of the bodies
    '''
    plt.rcParams['grid.color'] = 'dimgray'
    fig = plt.figure(figsize=(10,10), facecolor='black')
    ax = plt.gcf().add_subplot(111, projection='3d')
    ax.set_xlim([lim_inf[0], lim_sup[0]])
    ax.set_ylim([lim_inf[1], lim_sup[1]])
    ax.set_zlim([lim_inf[2], lim_sup[2]])
    ax.set_facecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('dimgray')
    ax.yaxis.pane.set_edgecolor('dimgray')
    ax.zaxis.pane.set_edgecolor('dimgray')
    colors_l=['yellow', 'orange', 'skyblue']
    colors=['gold','darkorange','cyan']
    for j in range(3):
        ax.plot(bodies[:,j,0], bodies[:,j,1],bodies[:,j,2], color=colors_l[j])#,'-', color=colors[j])
        ax.scatter(bodies[-1,j,0], bodies[-1,j,1], bodies[-1,j,2], marker='o', color=colors[j])
    ax.set_title(' Time : {:.3f} years'.format(time), color='dimgray')
    plt.gcf().savefig(image_folder+'image_{0:06}.png'.format(i))
    plt.close()

def create_video(image_folder='images/', video_name='my_video.mp4'):
    '''
    Creates a .mp4 video using the stored files images
    '''
    import os
    os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
    from os import listdir
    import moviepy.video.io.ImageSequenceClip
    fps = 15
    image_files = [image_folder+img for img in sorted(listdir(image_folder)) if img.endswith(".png")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(video_name)

if __name__=="__main__":
    '''
    Example of 3-Body system: Sun-Jupiter and Body_3
    '''
    #Planets information
    Body_1 = Sun # [x, y, z, vx, vy, vz, mass, orbital period]
    name1 = 'Sun'
    Body_2 = Jupiter # [x, y, z, vx, vy, vz, mass, orbital period]
    name2 = 'Jupiter'
    Body_3 = a_2021PH27 # [x, y, z, vx, vy, vz, mass, orbital period]
    name3 = 'a_2021PH27'
    
    
    # Creation of the time grid (in years)
    t = array([0.,max(Body_1[-1],Body_2[-1], Body_3[-1])])
    
    # Number of steps in the grid
    n = 10000
    
    # Arrays to store time information
    t1 = linspace(t[0], t[1], n)
    dt = (t[1]-t[0])/n
    print(f'dt={dt} and tf={t1[1]}')
    # Mass
    m1 = Body_1[6]
    m2 = Body_2[6]
    m3 = Body_3[6]
    mass = array([m1, m2, m3])
    
    # Initial Conditions
    q = zeros([n,3,6]) #  Motion information FR
    q[0, 0] = Body_1[:6] # initial x, y, vx, vy to particle 1
    q[0, 1] = Body_2[:6] # initial x, y, vx, vy to particle 2
    q[0, 2] = Body_3[:6] # initial x, y, vx, vy to particle 3


    # Solution to the problem using LeapFrog method
    q = PEFRL(Acceleration, q[0], mass, n, t)
    
    # Frequency at which .PNG images are written.
    img_step = 50
    # Folder to save the images
    image_folder = 'images/'
    # Name of the generated video
    video_name = 'my_video.mp4'
    # Center of the image
    center = array([0., 0., 0.])

    #create_images(q, dt, center, img_step, image_folder, video_name)
    #create_video(image_folder, video_name)
