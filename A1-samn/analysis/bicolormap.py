"""
BICOLORMAP

This program generators a two-color map, blue for negative, red for
positive changes, with grey in the middle. The input argument is how much
of a color gap there is between the red scale and the blue one.

The function has four parameters:
    gap: sets how big of a gap between red and blue color scales there is (0=no gap; 1=pure red and pure blue)
    mingreen: how much green to include at the extremes of the red-blue color scale
    redbluemix: how much red to mix with the blue and vice versa at the extremes of the scale
    epsilon: what fraction of the colormap to make gray in the middle

Examples:
    bicolormap(gap=0,mingreen=0,redbluemix=1,epsilon=0) # From pure red to pure blue with white in the middle
    bicolormap(gap=0,mingreen=0,redbluemix=0,epsilon=0.1) # Red -> yellow -> gray -> turquoise -> blue
    bicolormap(gap=0.3,mingreen=0.2,redbluemix=0,epsilon=0.01) # Red and blue with a sharp distinction between

Version: 2013sep13 by cliffk
"""

## Create colormap
def bicolormap(gap=0.1,mingreen=0.2,redbluemix=0.5,epsilon=0.01):
   from matplotlib.colors import LinearSegmentedColormap as makecolormap
   
   mng=mingreen; # Minimum amount of green to add into the colors
   mix=redbluemix; # How much red to mix with the blue an vice versa
   eps=epsilon; # How much of the center of the colormap to make gray
   omg=1-gap # omg = one minus gap
   
   cdict = {'red': ((0.00000, 0.0, 0.0),
                    (0.5-eps, mix, omg),
                    (0.50000, omg, omg),
                    (0.5+eps, omg, 1.0),
                    (1.00000, 1.0, 1.0)),

         'green':  ((0.00000, mng, mng),
                    (0.5-eps, omg, omg),
                    (0.50000, omg, omg),
                    (0.5+eps, omg, omg),
                    (1.00000, mng, mng)),

         'blue':   ((0.00000, 1.0, 1.0),
                    (0.5-eps, 1.0, omg),
                    (0.50000, omg, omg),
                    (0.5+eps, omg, mix),
                    (1.00000, 0.0, 0.0))}
   cmap = makecolormap('bicolormap',cdict,256)

   return cmap

## Show the staggering beauty of the color map's infinite possibilities
def testcolormap():
    from pylab import figure, subplot, imshow, colorbar, rand, show
    
    maps=[]
    maps.append(bicolormap()) # Default ,should work for most things
    maps.append(bicolormap(gap=0,mingreen=0,redbluemix=1,epsilon=0)) # From pure red to pure blue with white in the middle
    maps.append(bicolormap(gap=0,mingreen=0,redbluemix=0,epsilon=0.1)) # Red -> yellow -> gray -> turquoise -> blue
    maps.append(bicolormap(gap=0.3,mingreen=0.2,redbluemix=0,epsilon=0.01)) # Red and blue with a sharp distinction between
    nexamples=len(maps)
    
    figure(figsize=(5*nexamples,4))    
    for m in range(nexamples):
        subplot(1,nexamples,m+1)
        imshow(rand(20,20),cmap=maps[m],interpolation='none');
        colorbar()
    show()