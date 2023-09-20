"""
cfg.py 

Simulation configuration for A1 model (using NetPyNE)
This file has sim configs as well as specification for parameterized values in netParams.py 

Contributors: ericaygriffith@gmail.com, salvadordura@gmail.com
"""


from netpyne import specs
import pickle

cfg = specs.SimConfig()

#------------------------------------------------------------------------------
#
# SIMULATION CONFIGURATION
#
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Run parameters
#------------------------------------------------------------------------------
cfg.duration = 0.5*1e3			## Duration of the sim, in ms -- value from M1 cfg.py 
cfg.dt = 0.05                   ## Internal Integration Time Step -- value from M1 cfg.py 
cfg.verbose = 0                	## Show detailed messages
cfg.hParams['celsius'] = 37
cfg.createNEURONObj = 1
cfg.createPyStruct = 1
cfg.printRunTime = 0.1

cfg.connRandomSecFromList = False  # set to false for reproducibility 
cfg.cvode_active = False
cfg.cvode_atol = 1e-6
cfg.cache_efficient = True
cfg.oneSynPerNetcon = False
cfg.includeParamsLabel = False
cfg.printPopAvgRates = [0, cfg.duration]


#------------------------------------------------------------------------------
# Recording 
#------------------------------------------------------------------------------
cfg.allpops = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'PV4', 'SOM4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B', 'PV5B', 'SOM5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'IC']

alltypes = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'ITS4', 'PT5B', 'TC', 'HTC', 'IRE', 'TI']

cfg.recordTraces = {'V_soma': {'sec':'soma', 'loc':0.5, 'var':'v'}}  ## Dict with traces to record -- taken from M1 cfg.py 
cfg.recordStim = False			## Seen in M1 cfg.py
cfg.recordTime = False  		## SEen in M1 cfg.py 
cfg.recordStep = 1.0            ## Step size (in ms) to save data -- value from M1 cfg.py 

#cfg.recordLFP = [[-15, 500, 100]] #[[200, y, 200] for y in range(0, 2000, 400)]+[[200, 2500, 200], [200,2700,200]]

#------------------------------------------------------------------------------
# Saving
#------------------------------------------------------------------------------

cfg.simLabel = 'v28_sim1'
cfg.saveFolder = 'data/v28_manualTune/'                	## Set file output name
cfg.savePickle = False         	## Save pkl file
cfg.saveJson = True           	## Save json file
cfg.saveDataInclude = ['simData', 'simConfig', 'netParams'] ## seen in M1 cfg.py (line 58)
cfg.backupCfgFile = None 		## Seen in M1 cfg.py 
cfg.gatherOnlySimData = False	## Seen in M1 cfg.py 
cfg.saveCellSecs = True			## Seen in M1 cfg.py 
cfg.saveCellConns = 1		## Seen in M1 cfg.py 

#------------------------------------------------------------------------------
# Analysis and plotting 
#----------------------------------------------------------------------------- 

# cellGids = {'NGF1': 0, 'IT2': 45, 'SOM2': 146, 'PV2': 147, 'VIP2': 151, 'NGF2': 155, 'IT3': 158, 'SOM3': 1496, 'PV3': 1517, 'VIP3': 1569, 'NGF3': 1632, 'ITP4': 1677, 'ITS4': 1928, 'SOM4': 2179, 'PV4': 2186, 'VIP4': 2214, 'NGF4': 2218, 'IT5A': 2222, 'SOM5A': 2437, 'PV5A': 2450, 'VIP5A': 2472, 'NGF5A': 2475, 'IT5B': 2477, 'PT5B': 2689, 'SOM5B': 2901, 'PV5B': 2934, 'VIP5B': 2974, 'NGF5B': 2979, 'IT6': 2986, 'CT6': 3288, 'SOM6': 3590, 'PV6': 3609, 'VIP6': 3634, 'NGF6': 3637, 'TC': 3648, 'TCM': 3683, 'HTC': 3729, 'IRE': 3740, 'IREM': 3786}

# popGidRecord = [list(cellGids.values())[i] for i in [6,7,8,9,10,11,12,-1,-2,-3,-4,-5]]

cfg.analysis['plotTraces'] = {'include': [(pop, 0) for pop in cfg.allpops], 'oneFigPer': 'cell', 'overlay': True, 'saveFig': True, 'showFig': False, 'figSize':(12,8)} #[(pop,0) for pop in alltypes]		## Seen in M1 cfg.py (line 68) 

#cfg.analysis['plotRaster'] = {'include': cfg.allpops, 'saveFig': True, 'showFig': False, 'popRates': True, 'orderInverse': True, 'timeRange': [0,cfg.duration], 'figSize': (14,12), 'lw': 0.3, 'markerSize':10, 'marker': '.', 'dpi': 300}      	## Plot a raster
#cfg.analysis['plotLFP'] = {'plots': ['timeSeries', 'locations'], 'saveData': False, 'saveFig':1}
#cfg.analysis['plot2Dnet'] = True      	## Plot 2D visualization of cell positions & connections 


#------------------------------------------------------------------------------
# Cells
#------------------------------------------------------------------------------
cfg.weightNormThreshold = 5.0  # maximum weight normalization factor with respect to the soma

#------------------------------------------------------------------------------
# Synapses
#------------------------------------------------------------------------------
cfg.AMPATau2Factor = 1.0
cfg.synWeightFractionEE = [0.5, 0.5] # E->E AMPA to NMDA ratio
cfg.synWeightFractionEI = [0.5, 0.5] # E->I AMPA to NMDA ratio
cfg.synWeightFractionSOME = [0.9, 0.1] # SOM -> E GABAASlow to GABAB ratio
cfg.synWeightFractionNGF = [0.5, 0.5] # NGF GABAA to GABAB ratio


#------------------------------------------------------------------------------
# Network 
#------------------------------------------------------------------------------
## These values taken from M1 cfg.py (https://github.com/Neurosim-lab/netpyne/blob/development/examples/M1detailed/cfg.py)
cfg.singleCellPops = 1
cfg.singlePop = ''
cfg.removeWeightNorm = False
cfg.scale = 1.0     # Is this what should be used? 
cfg.sizeY = 2000.0 #1350.0 in M1_detailed # should this be set to 2000 since that is the full height of the column? 
cfg.sizeX = 400.0 # This may change depending on electrode radius 
cfg.sizeZ = 400.0
cfg.scaleDensity = 0.025 #0.075 # Should be 1.0 unless need lower cell density for test simulation or visualization


#------------------------------------------------------------------------------
# Connectivity
#------------------------------------------------------------------------------
cfg.synWeightFractionEE = [0.5, 0.5] # E->E AMPA to NMDA ratio
cfg.synWeightFractionEI = [0.5, 0.5] # E->I AMPA to NMDA ratio
cfg.synWeightFractionIE = [0.9, 0.1]  # SOM -> E GABAASlow to GABAB ratio (update this)
cfg.synWeightFractionII = [0.9, 0.1]  # SOM -> E GABAASlow to GABAB ratio (update this)

# Cortical
cfg.addConn = 0
cfg.EEGain = 1.0 
cfg.EIGain = 1.0 #0.75
cfg.IEGain = 1.0 #0.75
cfg.IIGain = 1.0 #0.5

## I->E/I layer weights (L2/3+4, L5, L6)
cfg.IEweights = [1.0, 1.0, 1.0] # [0.75, 0.75, 0.5]
cfg.IIweights = [1.5, 1.0, 1.0]

# Thalamic
cfg.addIntraThalamicConn = 0
cfg.addIntraThalamicConn = 0
cfg.addCorticoThalamicConn = 0
cfg.addThalamoCorticalConn = 0
#cfg.addMatrixThalamoCorticalConn = 1

cfg.intraThalamicGain = 1.0 #0.5
cfg.corticoThalamicGain = 1.0
cfg.thalamoCorticalGain = 1.0 #2.5
#cfg.matrixThalamoCorticalGain = 2.0

cfg.addSubConn = 0

#------------------------------------------------------------------------------
# Background inputs
#------------------------------------------------------------------------------
cfg.addBkgConn = 1
cfg.noiseBkg = 1.0  # firing rate random noise
cfg.delayBkg = 5.0  # (ms)
cfg.startBkg = 0  # start at 0 ms

# weights tuned to obtain low firing rates on each cell type with only bkg
cfg.weightBkg = 1
cfg.weightBkgE = 1
cfg.weightBkgI = 1

#  {'IT': 16.0, 'ITS4': 0.7, 'PT': 15.0, 'CT': 14.0,
#                 'PV': 28.0, 'SOM': 5.0, 'NGF': 65.0, 'VIP': 9.0,
#                 'TC': 1.0, 'HTC': 1.0, 'RE': 9.0, 'TI': 3.0}
cfg.rateBkg = {'exc': 40, 'inh': 40}


## options to provide external sensory input
cfg.randomThalInput = True  # provide random bkg inputs spikes (NetStim) to thalamic populations 

cfg.cochlearThalInput = False #{'numCells': 200, 'freqRange': [9*1e3, 11*1e3], 'toneFreq': 10*1e3, 'loudnessDBs': 50}  # parameters to generate realistic  auditory thalamic inputs using Brian Hears 

cfg.ICThalInput = None #{'file': 'data/ICoutput/ICoutput_CF_9600_10400_wav_01_ba_peter.mat', 'startTime': 500, 'weightE': 0.5, 'weightI': 0.5, 'probE': 0.12, 'probI': 0.26}}  # parameters to generate realistic cochlear + IC input ; weight =unitary connection somatic EPSP (mV)


#------------------------------------------------------------------------------
# Current inputs 
#------------------------------------------------------------------------------
cfg.addIClamp = 0

cfg.IClamp1 = {'pop': '', 'sec': 'soma', 'loc': 0.5, 'start': 100, 'dur': 400, 'amp': 0.4}

#------------------------------------------------------------------------------
# NetStim inputs 
#------------------------------------------------------------------------------

cfg.addNetStim = 0

## LAYER 1
cfg.NetStim1 = {'pop': 'VIP2', 'ynorm': [0,2.0], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 0.0, 'weight': 10.0, 'delay': 0}

# ## LAYER 2
# cfg.NetStim2 = {'pop': 'IT2',  'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}







