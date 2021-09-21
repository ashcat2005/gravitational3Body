def Plot_orbit(Evolution, Names, Central_Body=None, Outer_Body=None):
    '''
    ----------------------------------------------------------------
    Plot_orbit(Evolution):
    Draw the orbit of a system of bodies in the Central_Body frame.
    ----------------------------------------------------------------
    Arguments:
    Evolution: Numpy array with the cartesian position of each body 
        over a time interval.
    Names: Numpy array with the names of the bodies.
    Central_Body: Optional.
        Central Body. If it is None (default), then the central body
        is taken the first array, e.g, Evolution[:,0], otherwise the 
        central body is Evolution[:, Central_Body].
    Outer_Body: Optional.
        Body with the largest periapsis. If it is None (default), then
        It takes the second array, e.g, Evolution[:,1], otherwise the 
        Outer body is Evolution[:, Outer_Body].
    ----------------------------------------------------------------
    '''
    import plotly.graph_objects as go
    from numpy import amin, amax, ceil

    #Central Body
    if Central_Body is not None:
        j=Central_Body
    else:
        j=0

    #Outer Body
    if Outer_Body is not None:
        s=Outer_Body
    else:
        s=1

    # Limits for the plot
    boundary = max(abs(amin(Evolution[:,s,:3]-Evolution[:,j,:3])),abs(amax(Evolution[:,s,:3]-Evolution[:,j,:3])))*(1+0.1)
    
    # Interval
    dx = int(ceil(len(Evolution)/100000.))
    #Array of colors
    colors=['gold','darkorange','cyan','red']
    #Plot orbit
    fig = go.Figure()
    for i in range(len(Evolution[0])):
        if i==j:
            fig.add_trace(go.Scatter3d(x=Evolution[::dx,i,0]-Evolution[::dx,j,0],\
                                       y=Evolution[::dx,i,1]-Evolution[::dx,j,1],\
                                       z=Evolution[::dx,i,2]-Evolution[::dx,j,2],\
                                       mode='markers',marker=dict(size=5,color=colors[i%4]),name=Names[i]))
        else:
            fig.add_trace(go.Scatter3d(x=Evolution[::dx,i,0]-Evolution[::dx,j,0],\
                                       y=Evolution[::dx,i,1]-Evolution[::dx,j,1],\
                                       z=Evolution[::dx,i,2]-Evolution[::dx,j,2],\
                                       mode='markers',marker=dict(size=2,color=colors[i%4]),name=Names[i]))

    fig.update_layout(
        template="plotly_dark",
        scene = dict(
            xaxis = dict(nticks=5, range=[-boundary,boundary]),
            yaxis = dict(nticks=5, range=[-boundary,boundary]),
            zaxis = dict(nticks=5, range=[-boundary,boundary]),))
        
    fig.show()

def Plot_OrbElem(t, Orbital_elements, Name='test particle', Color='cyan'):
    '''
    ----------------------------------------------------------------
    Plot_OrbElem(t, Orbital_elements):
    Draw the orbital elements over a time interval.
    ----------------------------------------------------------------
    Arguments:
    t:  Time interval.
    Orbital_elements: Numpy array with the orbital elements of each body
        over a time interval, following the next format:
        Orbital_elements=[a,e,i,omega, Omega]
    Name: Body's name.
    Color: Plot's color. Crimson by default.
    ----------------------------------------------------------------
    '''
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    import matplotlib.ticker as mtick

    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.formatter.limits']= -3, 3
    fig, axs = plt.subplots(2, 3, figsize=(16,8))

    fig.suptitle('COEs of ' + Name,fontsize=20)

    #Plot Semi-major axis
    axs[0, 0].plot(t, Orbital_elements[:,0], color=Color)
    axs[0, 0].set_title('$a$')
    axs[0, 0].set_ylabel('[au]')
    axs[0, 0].set_xlabel('$t$ [yr]')

    #Plot Eccentricity
    axs[0, 1].plot(t, Orbital_elements[:,1], color=Color)
    axs[0, 1].set_title('$e$')
    axs[0, 1].set_xlabel('$t$ [yr]')

    #Plot Inclination
    axs[0, 2].plot(t, Orbital_elements[:,2], color=Color)
    axs[0, 2].set_title('$\iota$')
    axs[0, 2].set_ylabel('[rads]')
    axs[0, 2].set_xlabel('$t$ [yr]')

    #Plot argument of periapse
    axs[1, 0].plot(t, Orbital_elements[:,3], color=Color)
    axs[1, 0].set_title('$\omega$')
    axs[1, 0].set_ylabel('[rads]')
    axs[1, 0].set_xlabel('$t$ [yr]')

    #Plot Longitude of ascendind node
    axs[1, 1].plot(t, Orbital_elements[:, 4], color=Color)
    axs[1, 1].set_title('$\Omega$')
    axs[1, 1].set_ylabel('[rads]')
    axs[1, 1].set_xlabel('$t$ [yr]')

    plt.subplots_adjust(wspace=0.4,hspace=0.4)
    plt.setp(axs, xlim=[t[0],t[-1]])
                        
    axs[1, 0].set_position([0.24,0.125,0.228,0.343])
    axs[1, 1].set_position([0.55,0.125,0.228,0.343])
    axs[1, 2].set_visible(False)
    plt.show()
