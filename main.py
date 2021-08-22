from common import *

######### MAIN PROGRAM ########################################################


'''
Example of 2-Body system
'''
# Bodies initial data
sunM = 1.           # Solar masses
earthM = 3.00273e-6 # Solar masses
sun_ini_pos = array([0., 0., 0.])              # Initial position of the Sun
earth_ini_pos = array([1., 0., 0.])         # Initial position of the Earth
sun_ini_momentum = array([0., 0., 0.])         # Initial Momentum of the Sun
earth_ini_momentum = array([0.,1.898986e-5,0.]) # Initial Momentum of the Earth

masses = array([sunM, earthM])
ini_positions = array([sun_ini_pos, earth_ini_pos])
ini_momenta = array([sun_ini_momentum, earth_ini_momentum])
    
# Center of the image
center = array([0., 0., 0.])
# Limit for the plot
plot_limit = 2. #kpc
    
# Number of time-iterations executed by the program.
n = 5000 # Time steps
    
# Frequency at which .PNG images are written.
img_step = 50
# Folder to save the images
image_folder = 'images/'
# Name of the generated video
video_name = 'Sun-Earth.mp4'
    
bodies = system_init(masses, ini_positions, ini_momenta)
evolve(bodies, n, center, plot_limit, img_step, image_folder, video_name)
create_video(image_folder, video_name)


