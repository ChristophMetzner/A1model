"""
cfg.py 

Simulation configuration for A1 model (using NetPyNE)
This file has sim configs as well as specification for parameterized values in netParams.py 

Contributors: ericaygriffith@gmail.com, salvadordura@gmail.com, samnemo@gmail.com
"""


from netpyne import specs
import pickle
import numpy as np

cfg = specs.SimConfig()

#------------------------------------------------------------------------------
#
# SIMULATION CONFIGURATION
#
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Run parameters
#------------------------------------------------------------------------------
cfg.duration = 6e3             ## Duration of the sim, in ms
cfg.dt = 0.05                   ## Internal Integration Time Step
cfg.verbose = 0         	## Show detailed messages
cfg.hParams['celsius'] = 37
cfg.createNEURONObj = 1
cfg.createPyStruct = 1
cfg.printRunTime = 0.1

cfg.connRandomSecFromList = False  # set to false for reproducibility 
cfg.cvode_active = False
cfg.cvode_atol = 1e-6
cfg.cache_efficient = True
# cfg.printRunTime = 0.1  			## specified above 
cfg.oneSynPerNetcon = False
cfg.includeParamsLabel = False
cfg.printPopAvgRates = [0, cfg.duration]   # "printPopAvgRates": [[1500,1750],[1750,2000],[2000,2250],[2250,2500]]
cfg.validateNetParams = False

#------------------------------------------------------------------------------
# Recording 
#------------------------------------------------------------------------------
cfg.allpops = ['NGF1', 'IT2', 'SOM2', 'PV2', 'VIP2', 'NGF2', 'IT3',  'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'SOM4', 'PV4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'SOM5A', 'PV5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B',  'SOM5B', 'PV5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'SOM6', 'PV6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM', 'IC']
cfg.allCorticalPops = ['NGF1', 'IT2', 'SOM2', 'PV2', 'VIP2', 'NGF2', 'IT3',  'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'SOM4', 'PV4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'SOM5A', 'PV5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B',  'SOM5B', 'PV5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'SOM6', 'PV6', 'VIP6', 'NGF6']
cfg.allThalPops = ['TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM', 'IC']

alltypes = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'ITS4', 'PT5B', 'TC', 'HTC', 'IRE', 'TI']

cfg.recordTraces = {'V_soma': {'sec':'soma', 'loc': 0.5, 'var':'v'}}  ## Dict with traces to record -- taken from M1 cfg.py 
cfg.recordStim = False			## Seen in M1 cfg.py
cfg.recordTime = True  		## SEen in M1 cfg.py 
cfg.recordStep = 0.05            ## Step size (in ms) to save data -- value from M1 cfg.py 

cfg.recordLFP = [[100, y, 100] for y in range(0, 2000, 100)] #+[[100, 2500, 200], [100,2700,200]]			# null,
# cfg.recordLFP = [[x, 1000, 100] for x in range(100, 2200, 200)] #+[[100, 2500, 200], [100,2700,200]]
# cfg.saveLFPPops =  cfg.allCorticalPops #, "IT3", "SOM3", "PV3", "VIP3", "NGF3", "ITP4", "ITS4", "IT5A", "CT5A", "IT5B", "PT5B", "CT5B", "IT6", "CT6"]

cfg.recordDipole = True
# cfg.saveDipoleCells = ['all']
# cfg.saveDipolePops = cfg.allpops

#------------------------------------------------------------------------------
# Saving
#------------------------------------------------------------------------------

cfg.simLabel = 'ASSR_30_01_24_artFB_test_3'
cfg.saveFolder = 'data/' + cfg.simLabel  ## Set file output name
cfg.savePickle = True         							## Save pkl file
cfg.saveJson = False           							## Save json file
cfg.saveDataInclude = ['simData', 'simConfig', 'netParams', 'net'] 
cfg.backupCfgFile = None
cfg.gatherOnlySimData = False
cfg.saveCellSecs = True		   	 # False
cfg.saveCellConns = False

#------------------------------------------------------------------------------
# Analysis and plotting 
#----------------------------------------------------------------------------- 
#

#cfg.analysis['plotTraces'] = {'include': [(pop, 0) for pop in cfg.allpops], 'oneFigPer': 'trace', 'overlay': True, 'saveFig': True, 'showFig': False, 'figSize':(12,8)} #[(pop,0) for pop in alltypes]		## Seen in M1 cfg.py (line 68) 
#cfg.analysis['plotRaster'] = {'include': cfg.allpops, 'saveFig': True, 'showFig': False, 'popRates': True, 'orderInverse': True, 'timeRange': [0,cfg.duration], 'figSize': (14,12), 'lw': 0.3, 'markerSize': 3, 'marker': '.', 'dpi': 300}      	## Plot a raster
#cfg.analysis['plotSpikeStats'] = {'stats': ['rate'], 'figSize': (6,12), 'timeRange': [0, 2500], 'dpi': 300, 'showFig': 0, 'saveFig': 1}

#cfg.analysis['plotLFP'] = {'plots': ['timeSeries'], 'electrodes': [10], 'maxFreq': 80, 'figSize': (8,4), 'saveData': False, 'saveFig': True, 'showFig': False} # 'PSD', 'spectrogram'
#cfg.analysis['plotDipole'] = {'saveFig': True}
#cfg.analysis['plotEEG'] = {'saveFig': True}


#layer_bounds= {'L1': 100, 'L2': 160, 'L3': 950, 'L4': 1250, 'L5A': 1334, 'L5B': 1550, 'L6': 2000}
#cfg.analysis['plotCSD'] = {'spacing_um': 100, 'LFP_overlay': 1, 'layer_lines': 1, 'layer_bounds': layer_bounds, 'saveFig': 1, 'showFig': 0}
#cfg.analysis['plot2Dnet'] = True      	## Plot 2D visualization of cell positions & connections 


#------------------------------------------------------------------------------
# Cells
#------------------------------------------------------------------------------
cfg.weightNormThreshold = 5.0  # maximum weight normalization factor with respect to the soma
cfg.weightNormScaling = {'NGF_reduced': 1.0, 'ITS4_reduced': 1.0}
cfg.ihGbar = 1.0 
cfg.KgbarFactor = 1.0
# For testing reduction in T-type calcium channel conductances
# cfg.tTypeCorticalFactor = 1.0 
# cfg.tTypeThalamicFactor = 1.0 
# For testing NMDAR manipulations:
# cfg.NMDARfactor = 1.0

#------------------------------------------------------------------------------
# Synapses
#------------------------------------------------------------------------------
cfg.AMPATau2Factor = 1.0
cfg.synWeightFractionEE = [0.5, 0.5] # E->E AMPA to NMDA ratio
cfg.synWeightFractionEI = [0.5, 0.5] # E->I AMPA to NMDA ratio
cfg.synWeightFractionEI_CustomCort = [0.5, 0.5] # E->I AMPA to NMDA ratio custom for cortex NMDA manipulation
cfg.synWeightFractionSOME = [0.9, 0.1] # SOM -> E GABAASlow to GABAB ratio
cfg.synWeightFractionNGF = [0.5, 0.5] # NGF GABAA to GABAB ratio
cfg.synWeightFractionENGF = [0.834, 0.166] # NGF AMPA to NMDA ratio
cfg.useHScale = False

# cfg.synWeightFractionIE = [0.9, 0.1]
# cfg.synWeightFractionII = [0.9, 0.1]


#------------------------------------------------------------------------------
# Network 
#------------------------------------------------------------------------------
## These values taken from M1 cfg.py (https://github.com/Neurosim-lab/netpyne/blob/development/examples/M1detailed/cfg.py)
cfg.singleCellPops = False #True #False
cfg.singlePop = ''
cfg.removeWeightNorm = False
cfg.scale = 1.0     # Is this what should be used? 
cfg.sizeY = 2000.0 #1350.0 in M1_detailed # should this be set to 2000 since that is the full height of the column? 
cfg.sizeX = 200.0 # 400 - This may change depending on electrode radius 
cfg.sizeZ = 200.0
cfg.scaleDensity = 1.0 #0.25 #1.0 #0.075 # Should be 1.0 unless need lower cell density for test simulation or visualization


#------------------------------------------------------------------------------
# Connectivity
#------------------------------------------------------------------------------
cfg.synWeightFractionIE = [0.9, 0.1]  # SOM -> E GABAASlow to GABAB ratio (update this)
cfg.synWeightFractionII = [0.9, 0.1]  # SOM -> E GABAASlow to GABAB ratio (update this)

# Cortical
cfg.addConn = 1

cfg.EEGain = 1.0
cfg.EIGain = 1.0 # 1.8600534795309025 	
cfg.IEGain = 1.0 #0.75
cfg.IIGain = 1.0 #0.5

## E/I->E/I layer weights (L1-3, L4, L5, L6)
cfg.EELayerGain = {'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0 , '5A': 1.0, '5B': 1.0, '6': 1.0}
cfg.EILayerGain = {'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0 , '5A': 1.0, '5B': 1.0, '6': 1.0}
cfg.IELayerGain = {'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0 , '5A': 1.0, '5B': 1.0, '6': 1.0}
cfg.IILayerGain = {'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0 , '5A': 1.0, '5B': 1.0, '6': 1.0}



# E -> E based on postsynaptic cortical E neuron population
# cfg.EEPopGain = {'IT2': 1.3125, 'IT3': 1.55, 'ITP4': 1.0, 'ITS4': 1.0, 'IT5A': 1.0, 'CT5A': 1.1, 'IT5B': 0.375, 'CT5B': 1.1, 'PT5B': 1.0, 'IT6': 1.0, 'CT6': 1.0}
#cfg.EEPopGain = {'IT2': 1.3125, 'IT3': 1.55, 'ITP4': 1.0, 'ITS4': 1.0, 'IT5A': 1.05, 'CT5A': 1.1500000000000001, 'IT5B': 0.425, 'CT5B': 1.1500000000000001, 'PT5B': 1.05, 'IT6': 1.05, 'CT6': 1.05} # this is from after generation 203 of optunaERP_23dec23_ , values used in
cfg.EEPopGain = {'IT2': 2.5, 'IT3': 1.75, 'ITP4': 1.0, 'ITS4': 1.0, 'IT5A': 1.05, 'CT5A': 1.1500000000000001, 'IT5B': 0.425, 'CT5B': 1.1500000000000001, 'PT5B': 1.05, 'IT6': 1.05, 'CT6': 1.05} # this is from after generation 203 of optunaERP_23dec23_ , values used in
#cfg.EEPopGain = {'IT2': 1.0, 'IT3': 1.0, 'ITP4': 1.0, 'ITS4': 1.0, 'IT5A': 1.0, 'CT5A': 1.0, 'IT5B': 1.0, 'CT5B': 1.0, 'PT5B': 1.0, 'IT6': 1.0, 'CT6': 1.0} # default


# gains from E -> I based on postsynaptic cortical I neuron population
#cfg.EIPopGain = {'NGF1': 1.0, 'SOM2': 1.0, 'PV2': 1.0, 'VIP2': 1.0, 'NGF2': 1.0, 'SOM3': 1.0, 'PV3': 1.0, 'VIP3': 1.0, 'NGF3': 1.0, 'SOM4': 1.0, 'PV4': 1.0, 'VIP4': 1.0, 'NGF4': 1.0, 'SOM5A': 1.0, 'PV5A': 1.4, 'VIP5A': 1.25, 'NGF5A': 0.8, 'SOM5B': 1.0, 'PV5B': 1.4, 'VIP5B': 1.4, 'NGF5B': 0.9, 'SOM6': 1.0, 'PV6': 1.4, 'VIP6': 1.4, 'NGF6': 0.65}
#cfg.EIPopGain = {'NGF1': 1.0, 'SOM2': 1.0, 'PV2': 1.0, 'VIP2': 1.0, 'NGF2': 1.0, 'SOM3': 1.0, 'PV3': 0.5, 'VIP3': 1.0, 'NGF3': 1.0, 'SOM4': 1.0, 'PV4': 0.5, 'VIP4': 1.0, 'NGF4': 1.0, 'SOM5A': 1.0, 'PV5A': 1.4, 'VIP5A': 1.25, 'NGF5A': 0.8, 'SOM5B': 1.0, 'PV5B': 1.45, 'VIP5B': 1.4, 'NGF5B': 0.9500000000000001, 'SOM6': 1.0, 'PV6': 1.4, 'VIP6': 1.3499999999999999, 'NGF6': 0.65} # this is from after generation 203 of optunaERP_23dec23_ , values used in generation 204 of the same optimization
cfg.EIPopGain = {'NGF1': 1.0, 'SOM2': 1.0, 'PV2': 1.0, 'VIP2': 1.0, 'NGF2': 0.1, 'SOM3': 0.25, 'PV3': 0.15, 'VIP3': 0.25, 'NGF3': 0.1, 'SOM4': 0.5, 'PV4': 0.5, 'VIP4': 1.0, 'NGF4': 0.1, 'SOM5A': 1.0, 'PV5A': 1.0, 'VIP5A': 1.0, 'NGF5A': 1.0, 'SOM5B': 1.0, 'PV5B': 1.0, 'VIP5B': 1.0, 'NGF5B': 1.0, 'SOM6': 1.0, 'PV6': 1.0, 'VIP6': 1.0, 'NGF6': 1.0} 
#cfg.EIPopGain = {'NGF1': 1.0, 'SOM2': 1.0, 'PV2': 1.0, 'VIP2': 1.0, 'NGF2': 1.0, 'SOM3': 1.0, 'PV3': 1.0, 'VIP3': 1.0, 'NGF3': 1.0, 'SOM4': 1.0, 'PV4': 1.0, 'VIP4': 1.0, 'NGF4': 1.0, 'SOM5A': 1.0, 'PV5A': 1.0, 'VIP5A': 1.0, 'NGF5A': 1.0, 'SOM5B': 1.0, 'PV5B': 1.0, 'VIP5B': 1.0, 'NGF5B': 1.0, 'SOM6': 1.0, 'PV6': 1.0, 'VIP6': 1.0, 'NGF6': 1.0} # default


## E->I by target cell type
cfg.EICellTypeGain= {'PV': 1.0, 'SOM': 1.0, 'VIP': 1.0, 'NGF': 0.1} # default: {'PV': 1.0, 'SOM': 1.0, 'VIP': 1.0, 'NGF': 1.0}

## I->E by target cell type
cfg.IECellTypeGain= {'PV': 1.0, 'SOM': 1.0, 'VIP': 1.0, 'NGF': 1.0}


# Thalamic
cfg.addIntraThalamicConn = 1.0
cfg.addCorticoThalamicConn = 1.0
cfg.addThalamoCorticalConn = 1.0

cfg.thalamoCorticalGain = 1.0
cfg.intraThalamicGain = 1.0
cfg.corticoThalamicGain = 1.0

# these params control IC -> Thalamic Core
cfg.ICThalweightECore = 2.5 # 1.0
cfg.ICThalweightICore = 0.75 # 0.25
cfg.ICThalprobECore = 0.16
cfg.ICThalprobICore = 0.09

# these params control IC -> Thalamic Matrix
cfg.ICThalMatrixCoreFactor = 0.08 #0.1
cfg.ICThalweightEMatrix = cfg.ICThalweightECore * cfg.ICThalMatrixCoreFactor
cfg.ICThalweightIMatrix = cfg.ICThalweightICore * cfg.ICThalMatrixCoreFactor
cfg.ICThalprobEMatrix = cfg.ICThalprobECore 
cfg.ICThalprobIMatrix = cfg.ICThalprobICore 

# these params added from Christoph Metzner branch
cfg.thalL4PV = 0.075 # 0.5 
cfg.thalL4SOM = 0.25 # 0.5 
cfg.thalL4E = 2.5 #1.0 

cfg.thalL4VIP = 1.0
cfg.thalL4NGF = 1.0

cfg.L4L3E    = 2.0 # 1.0
cfg.L4L3PV   = 0.1
cfg.L4L3SOM  = 1.0

cfg.L4L3VIP = 0.1
cfg.L4L3NGF = 1.0

cfg.artFB = 1
cfg.artFBweight = 5.0
cfg.artFBprob = 0.5

cfg.addSubConn = 1


## full weight conn matrix
with open('conn/conn.pkl', 'rb') as fileObj: connData = pickle.load(fileObj)
cfg.wmat = connData['wmat']


#------------------------------------------------------------------------------
# Background inputs
#------------------------------------------------------------------------------
cfg.addBkgConn = 1
cfg.noiseBkg = 1.0  # firing rate random noise
cfg.delayBkg = 5.0  # (ms)
cfg.startBkg = 0  # start at 0 ms

# cfg.weightBkg = {'IT': 12.0, 'ITS4': 0.7, 'PT': 14.0, 'CT': 14.0,
#                 'PV': 28.0, 'SOM': 5.0, 'NGF': 80.0, 'VIP': 9.0,
#                 'TC': 1.8, 'HTC': 1.55, 'RE': 9.0, 'TI': 3.6}
cfg.rateBkg = {'exc': 40, 'inh': 40}

## options to provide external sensory input
#cfg.randomThalInput = True  # provide random bkg inputs spikes (NetStim) to thalamic populations 

cfg.EbkgThalamicGain = 4.0
cfg.IbkgThalamicGain = 4.0

cfg.cochlearThalInput = False #{'numCells': 200, 'freqRange': [9*1e3, 11*1e3], 'toneFreq': 10*1e3, 'loudnessDBs': 50}  # parameters to generate realistic  auditory thalamic inputs using Brian Hears

#------------------------------------------------------------------------------
# Current inputs 
#------------------------------------------------------------------------------
cfg.addIClamp = 0

#------------------------------------------------------------------------------
# NetStim inputs 
#------------------------------------------------------------------------------

cfg.addNetStim = 0 #1

## LAYER 1
# cfg.NetStim1 = {'pop': 'NGF1', 'ynorm': [0,2.0], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 0.0, 'weight': 10.0, 'delay': 0}

# ## LAYER 2
# cfg.NetStim2 = {'pop': 'IT2',  'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}

## LAYER 3
#cfg.NetStim3 = {'pop': 'IT3',  'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/20.0, 'noise': 0.0, 'number': 20.0,   'weight': 10.0, 'delay': 0}


cfg.tune = {}


# ------------------------ ADD PARAM VALUES FROM .JSON FILES: 
# COMMENT THIS OUT IF USING GCP !!! ONLY USE IF USING NEUROSIM!!! 
import json

with open('data/v34_batch25/trial_2142/trial_2142_cfg.json', 'rb') as f:       # 'data/salva_runs/v29_batch3_trial_13425_cfg.json'
	cfgLoad = json.load(f)['simConfig']


## UPDATE CORTICAL GAIN PARAMS 
cfg.EEGain = cfgLoad['EEGain']
cfg.EIGain = cfgLoad['EIGain']
cfg.IEGain = cfgLoad['IEGain']
cfg.IIGain = cfgLoad['IIGain']

cfg.EICellTypeGain['PV'] =  cfgLoad['EICellTypeGain']['PV']
cfg.EICellTypeGain['SOM'] = cfgLoad['EICellTypeGain']['SOM']
cfg.EICellTypeGain['VIP'] = cfgLoad['EICellTypeGain']['VIP']
cfg.EICellTypeGain['NGF'] = cfgLoad['EICellTypeGain']['NGF']

cfg.IECellTypeGain['PV'] = cfgLoad['IECellTypeGain']['PV']
cfg.IECellTypeGain['SOM'] = cfgLoad['IECellTypeGain']['SOM']
cfg.IECellTypeGain['VIP'] = cfgLoad['IECellTypeGain']['VIP']
cfg.IECellTypeGain['NGF'] = cfgLoad['IECellTypeGain']['NGF']

cfg.EILayerGain['1'] = cfgLoad['EILayerGain']['1']
cfg.IILayerGain['1'] = cfgLoad['IILayerGain']['1']

cfg.EELayerGain['2'] = cfgLoad['EELayerGain']['2']
cfg.EILayerGain['2'] = cfgLoad['EILayerGain']['2']
cfg.IELayerGain['2'] = cfgLoad['IELayerGain']['2']
cfg.IILayerGain['2'] = cfgLoad['IILayerGain']['2']


cfg.EELayerGain['3'] = cfgLoad['EELayerGain']['3']
cfg.EILayerGain['3'] = cfgLoad['EILayerGain']['3']
cfg.IELayerGain['3'] = cfgLoad['IELayerGain']['3']
cfg.IILayerGain['3'] = cfgLoad['IILayerGain']['3']


cfg.EELayerGain['4'] = cfgLoad['EELayerGain']['4']
cfg.EILayerGain['4'] = cfgLoad['EILayerGain']['4']
cfg.IELayerGain['4'] = cfgLoad['IELayerGain']['4']
cfg.IILayerGain['4'] = cfgLoad['IILayerGain']['4']

cfg.EELayerGain['5A'] = cfgLoad['EELayerGain']['5A']
cfg.EILayerGain['5A'] = cfgLoad['EILayerGain']['5A']
cfg.IELayerGain['5A'] = cfgLoad['IELayerGain']['5A']
cfg.IILayerGain['5A'] = cfgLoad['IILayerGain']['5A']

cfg.EELayerGain['5B'] = cfgLoad['EELayerGain']['5B']
cfg.EILayerGain['5B'] = cfgLoad['EILayerGain']['5B']
cfg.IELayerGain['5B'] = cfgLoad['IELayerGain']['5B'] 
cfg.IILayerGain['5B'] = cfgLoad['IILayerGain']['5B']

cfg.EELayerGain['6'] = cfgLoad['EELayerGain']['6']  
cfg.EILayerGain['6'] = cfgLoad['EILayerGain']['6']  
cfg.IELayerGain['6'] = cfgLoad['IELayerGain']['6']  
cfg.IILayerGain['6'] = cfgLoad['IILayerGain']['6']  



# UPDATE THALAMIC GAIN PARAMS
cfg.thalamoCorticalGain = cfgLoad['thalamoCorticalGain']
cfg.intraThalamicGain = cfgLoad['intraThalamicGain']
cfg.EbkgThalamicGain = cfgLoad['EbkgThalamicGain']
cfg.IbkgThalamicGain = cfgLoad['IbkgThalamicGain']



# UPDATE WMAT VALUES
cfg.wmat = cfgLoad['wmat']


cfg.ICThalInput = {'file': 'data/ICoutput/40Hz_10kHz_4s_AM_click_train_1kBMF_100CF.mat', #'data/ICoutput/ICoutput_CF_5256_6056_wav_BBN_100ms_burst.mat', # BBN_trials/ICoutput_CF_9600_10400_wav_BBN_100ms_burst_AN.mat', 
                   'startTime': 1500, #list(np.arange(4000, 8000, 300)),
                   'weightECore': cfg.ICThalweightECore,
                   'weightICore': cfg.ICThalweightICore,
                   'probECore': cfg.ICThalprobECore, 
                   'probICore': cfg.ICThalprobICore,
                   'weightEMatrix': cfg.ICThalweightEMatrix,
                   'weightIMatrix': cfg.ICThalweightIMatrix,
                   'probEMatrix': cfg.ICThalprobEMatrix,
                   'probIMatrix': cfg.ICThalprobIMatrix,
                   'seed': 1}  # SHOULD THIS BE ZERO?   

cfg.artFB = {'file':'data/FBinput/FBinput_test_numCell200.pkl',
             'weight': cfg.artFBweight,
             'prob': cfg.artFBprob,
            }                   
                   
#cfg.ICThalInput = {'file': 'data/ICoutput/SAM_Tone_input_40Hz.mat', #'data/ICoutput/ICoutput_CF_5256_6056_wav_BBN_100ms_burst.mat', # BBN_trials/ICoutput_CF_9600_10400_wav_BBN_100ms_burst_AN.mat', 
#                   'startTime': 1500, #list(np.arange(4000, 8000, 300)),
#                   'weightECore': cfg.ICThalweightECore,
#                   'weightICore': cfg.ICThalweightICore,
#                   'probECore': cfg.ICThalprobECore, 
#                   'probICore': cfg.ICThalprobICore,
#                   'weightEMatrix': cfg.ICThalweightEMatrix,
#                   'weightIMatrix': cfg.ICThalweightIMatrix,
#                   'probEMatrix': cfg.ICThalprobEMatrix,
#                   'probIMatrix': cfg.ICThalprobIMatrix,
#                   'seed': 1}  # SHOULD THIS BE ZERO?                   
