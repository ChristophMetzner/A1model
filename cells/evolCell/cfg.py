"""
cfg.py 

Simulation configuration 

Contributors: salvadordura@gmail.com
"""

import numpy as np
from netpyne import specs

cfg = specs.SimConfig()

#------------------------------------------------------------------------------
#
# SIMULATION CONFIGURATION
#
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Run parameters
#------------------------------------------------------------------------------
cfg.duration = 10*1e3			## Duration of the sim, in ms -- value from M1 cfg.py 
cfg.dt = 0.05                   ## Internal Integration Time Step -- value from M1 cfg.py 
cfg.verbose = 0           	## Show detailed messages
cfg.hParams['celsius'] = 37
cfg.printRunTime = 1
cfg.printPopAvgRates = [500, cfg.duration]


#------------------------------------------------------------------------------
# Recording 
#------------------------------------------------------------------------------
cfg.recordTraces = {'V_soma': {'sec': 'soma', 'loc': 0.5, 'var': 'v'}}
cfg.recordStims = False  
cfg.recordStep = 0.1 


#------------------------------------------------------------------------------
# Saving
#------------------------------------------------------------------------------
cfg.simLabel = 'sim2'
cfg.saveFolder = 'data'
cfg.savePickle = True
cfg.saveJson = False
cfg.saveDataInclude = ['simConfig', 'netParams', 'net', 'simData']


#------------------------------------------------------------------------------
# Analysis and plotting 
#------------------------------------------------------------------------------
cfg.analysis['plotfI'] = {'amps': [0.1], 'times': [1000], 'dur': 1000, 'targetRates': [10], 'saveFig': True, 'showFig': True}

cfg.analysis['plotTraces'] = {'include': [0], 'oneFigPer': 'cell', 'saveFig': True, 
 							  'showFig': False, 'figSize': (30,4), 'timeRange': [0,cfg.duration]}

#------------------------------------------------------------------------------
# Current inputs 
#------------------------------------------------------------------------------
cfg.addIClamp = 1

cfg.IClamp1 = {'pop': 'NGF', 'sec': 'soma', 'loc': 0.5, 'start': 100, 'dur': 1000, 'amp': 0.5}


#------------------------------------------------------------------------------
# NetStim inputs 
#------------------------------------------------------------------------------

cfg.addNetStim = 1

cfg.NetStim1 = {'pop': 'NGF', 'ynorm': [0,2.0], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 0.0, 'weight': 10.0, 'delay': 0}

#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------
# example of how to set params; but set from batch.py
cfg.tune = specs.Dict()
'''
cfg.tune['soma']['vinit'] = -75.0
cfg.tune['dend']['cadad']['kd'] = 0.0
cfg.tune['dend1']['Ra'] = 0.015915494309189534
'''