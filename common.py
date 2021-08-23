# Supernova Code

from copy import deepcopy
from numpy import array, ones, empty, random, sqrt, pi
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D


##### Simulation Parameters ########################################################

# Gravitational constant in units of [au^3 M_sun^-1 yr-2]
G = 4.*pi**2

# Discrete time step.
dt = 1.e-3 # [yr]

####################################################################################

class Body:
    '''
    Each Body object will represent a particle.
    '''
    def __init__(self, m, position, momentum):
        '''
        Creates a particle using the attributes
        .mass : scalar
        .position : NumPy array  with the coordinates [x,y,z]
        .momentum : NumPy array  with the components [px,py,pz]
        '''
        self.m = m
        self.m_pos = m * position
        self.momentum = momentum
    
    def position(self):
        '''
        Returns the physical coordinates of the Body.
        '''
        return self.m_pos / self.m


def distance_between(body1, body2):
    '''
    Returns the distance between body1 and body2.
    '''
    return norm(body1.position() - body2.position())


def gravitational_force(body1, body2):
    '''
    Returns the gravitational force that body2 exerts on body1.
    A short distance cutoff is introduced in order to avoid numerical
    divergences in the gravitational force.
    '''
    cutoff_dist = 2.e-4
    d = distance_between(body1, body2)
    if d < cutoff_dist:
        #print('Collision!')
        # Returns no Force
        return array([0., 0., 0.])
    else:
        # Gravitational force
        return -G*body1.m*body2.m*(body1.position() - body2.position())/d**3


def force_on(particle, other_bodies):
    '''
    Returns the total force on the particle due to the other_bodies
    '''
    return sum(gravitational_force(particle, body) for body in other_bodies)


def verlet(particles, other_bodies, dt):
    '''
    Verlet method for time evolution.
    '''
    for p in particles:
        force = force_on(p, other_bodies)
        p.momentum += force*dt
        p.m_pos += p.momentum*dt


def system_init(masses, ini_positions, ini_momenta):
    '''
    This function initialize the N-body system by creating the objects of the Body class
    '''
    bodies = []
    for i in range(len(masses)):
        bodies.append(Body(masses[i], ini_positions[i], ini_momenta[i]))
    return bodies



def evolve(particles, n, center, plot_limit, img_step, image_folder='images/', video_name='my_video.mp4'):
    '''
    This function evolves the system in time using the Verlet algorithm
    '''
    # Limits for the axes in the plot
    axis_limit = 1.5*plot_limit
    lim_inf = [center[0]-axis_limit, center[1]-axis_limit, center[2]-axis_limit]
    lim_sup = [center[0]+axis_limit, center[1]+axis_limit, center[2]+axis_limit]
    # Principal loop over time iterations.
    for i in range(n+1):
        for p in particles:
            verlet(particles, particles, dt)
        #Write the image files
        if i%img_step==0:
            print("Writing image at iteration {0}".format(i))
            plot_bodies(particles, i//img_step, 2*i*dt, lim_inf, lim_sup, image_folder)


def plot_bodies(bodies, i, time, lim_inf, lim_sup, image_folder='images/'):
    '''
    Writes an image file with the current position of the bodies
    '''
    plt.rcParams['grid.color'] = 'dimgray'
    #plt.rcParams['axes.edgecolor'] = 'dimgray'
    #plt.rcParams['axes.labelcolor'] = 'dimgray'
    fig = plt.figure(figsize=(10,10), facecolor='black')
    ax = plt.gcf().add_subplot(111, projection='3d')
    ax.set_xlim([lim_inf[0], lim_sup[0]])
    ax.set_ylim([lim_inf[1], lim_sup[1]])
    ax.set_zlim([lim_inf[2], lim_sup[2]])
    #ax.set_proj_type('ortho')
    ax.set_facecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('dimgray')
    ax.yaxis.pane.set_edgecolor('dimgray')
    ax.zaxis.pane.set_edgecolor('dimgray')
    #ax.xaxis.label.set_color('dimgray')
    #ax.yaxis.label.set_color('dimgray')
    #ax.zaxis.label.set_color('dimgray')
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')
    #ax.grid(False)
    for body in bodies:
        pos = body.position()
        ax.scatter(pos[0], pos[1], pos[2], marker='o', color='lightcyan')
    ax.set_title(' Time : {:.3f} years'.format(time), color='dimgray')
    plt.gcf().savefig(image_folder+'image_{0:06}.png'.format(i))
    plt.close()


def create_video(image_folder='images/', video_name='my_video.mp4'):
    '''
    Creates a .mp4 video using the stored files images
    '''
    from os import listdir
    import moviepy.video.io.ImageSequenceClip
    fps = 15
    image_files = [image_folder+img for img in sorted(listdir(image_folder)) if img.endswith(".png")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(video_name)




if __name__=="__main__":
    '''
    Example of 2-Body system: Sun-Earth
    '''
    # Bodies initial data
    sunM = 1.           # Solar masses
    earthM = 3.00273e-6 # Solar masses
    sun_ini_pos = array([0., 0., 0.])              # Initial position of the Sun [au]
    earth_ini_pos = array([1., 0., 0.])         # Initial position of the Earth [au]
    sun_ini_momentum = array([0., 0., 0.])         # Initial Momentum of the Sun [au/yr]
    earth_ini_momentum = array([0.,1.898986e-5,0.]) # Initial Momentum of the Earth [au/yr]
    
    masses = array([sunM, earthM])
    ini_positions = array([sun_ini_pos, earth_ini_pos])
    ini_momenta = array([sun_ini_momentum, earth_ini_momentum])
    
    # Center of the image
    center = array([0., 0., 0.])
    # Limit for the plot
    plot_limit = 2. # [au]
    
    # Number of time-iterations executed by the program.
    n = 5000 # Time steps
    
    # Frequency at which .PNG images are written.
    img_step = 50
    # Folder to save the images
    image_folder = 'images/'
    # Name of the generated video
    video_name = 'my_video.mp4'
    
    bodies = system_init(masses, ini_positions, ini_momenta)
    evolve(bodies, n, center, plot_limit, img_step, image_folder, video_name)
    create_video(image_folder, video_name)
