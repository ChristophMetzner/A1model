from netpyne import specs, sim
from neuron import h
from collections import OrderedDict
import pickle, json, os, time
import numpy as np
from conf import dconf # configuration dictionary

"""
cfg.py 

Simulation configuration for A1 model (using NetPyNE)
This file has sim configs as well as specification for parameterized values in netParams.py 

Contributors: ericaygriffith@gmail.com, salvadordura@gmail.com, samnemo@gmail.com , & Christoph Metzner
"""

cfg = specs.SimConfig()

#------------------------------------------------------------------------------
#
# SIMULATION CONFIGURATION
#
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Run parameters
#------------------------------------------------------------------------------
cfg.duration = dconf['sim']['duration'] # Duration of the sim, in ms
cfg.dt = dconf['sim']['dt']                  ## Internal Integration Time Step
cfg.verbose = dconf['verbose']         	## Show detailed messages
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
cfg.allpops = ['NGF1', 'IT2', 'SOM2', 'PV2', 'VIP2', 'NGF2', 'IT3',  'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'SOM4', 'PV4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'SOM5A', 'PV5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B',  'SOM5B', 'PV5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'SOM6', 'PV6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM', 'IC', 'cochlea']
cfg.allCorticalPops = ['NGF1', 'IT2', 'SOM2', 'PV2', 'VIP2', 'NGF2', 'IT3',  'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'SOM4', 'PV4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'SOM5A', 'PV5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B',  'SOM5B', 'PV5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'SOM6', 'PV6', 'VIP6', 'NGF6']
cfg.allThalPops = ['TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM', 'IC']

alltypes = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'ITS4', 'PT5B', 'TC', 'HTC', 'IRE', 'TI']

cfg.recordTraces = {'V_soma': {'sec':'soma', 'loc': 0.5, 'var':'v'}}  ## Dict with traces to record -- taken from M1 cfg.py 
cfg.recordStim = False			## Seen in M1 cfg.py
cfg.recordTime = True  		## SEen in M1 cfg.py 
cfg.recordStep = dconf['sim']['recordStep']            ## Step size (in ms) to save data -- value from M1 cfg.py 

cfg.recordLFP = [[100, y, 100] for y in range(0, 2000, 100)] #+[[100, 2500, 200], [100,2700,200]]			# null,
# cfg.recordLFP = [[x, 1000, 100] for x in range(100, 2200, 200)] #+[[100, 2500, 200], [100,2700,200]]
# cfg.saveLFPPops =  cfg.allCorticalPops #, "IT3", "SOM3", "PV3", "VIP3", "NGF3", "ITP4", "ITS4", "IT5A", "CT5A", "IT5B", "PT5B", "CT5B", "IT6", "CT6"]

# cfg.recordDipole = True
# cfg.saveDipoleCells = ['all']
# cfg.saveDipolePops = cfg.allpops

#------------------------------------------------------------------------------
# Saving
#------------------------------------------------------------------------------

cfg.simLabel = dconf['sim']['name']
cfg.saveFolder = 'data/' + cfg.simLabel  ## Set file output name
cfg.savePickle = True         							## Save pkl file
cfg.saveJson = False           							## Save json file
cfg.saveDataInclude = ['simData', 'simConfig', 'netParams', 'net'] 
cfg.backupCfgFile = None
cfg.gatherOnlySimData = False
cfg.saveCellSecs = dconf['sim']['saveCellSecs']
cfg.saveCellConns = dconf['sim']['saveCellConns']

#------------------------------------------------------------------------------
# Analysis and plotting 
#----------------------------------------------------------------------------- 
#

cfg.analysis['plotTraces'] = {'include': [(pop, 0) for pop in cfg.allpops], 'oneFigPer': 'trace', 'overlay': True, 'saveFig': True, 'showFig': False, 'figSize':(12,8)} #[(pop,0) for pop in alltypes]		## Seen in M1 cfg.py (line 68) 
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
cfg.weightNormThreshold = 5.0  # maximum weight normalization factor with respect to the somaw
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
cfg.synWeightFractionEE = dconf['syn']['synWeightFractionEE'] # [0.5, 0.5] # E->E AMPA to NMDA ratio
cfg.synWeightFractionEI = dconf['syn']['synWeightFractionEI'] # [0.5, 0.5] # E->I AMPA to NMDA ratio
cfg.synWeightFractionEI_CustomCort = dconf['syn']['synWeightFractionEI_CustomCort'] # [0.5, 0.5] # E->I AMPA to NMDA ratio custom for cortex NMDA manipulation
cfg.synWeightFractionNGF = dconf['syn']['synWeightFractionNGF'] # {'E':[0.5, 0.5],'I':[0.9,0.1]} # NGF GABAA to GABAB ratio when connecting to E , I neurons
cfg.synWeightFractionSOM = dconf['syn']['synWeightFractionSOM'] # {'E':[0.9, 0.1],'I':[0.9,0.1]} # SOM GABAA to GABAB ratio when connection to E , I neurons
cfg.synWeightFractionENGF = dconf['syn']['synWeightFractionENGF'] # [0.834, 0.166] # AMPA to NMDA ratio for E -> NGF connections
cfg.useHScale = False

#------------------------------------------------------------------------------
# Network 
#------------------------------------------------------------------------------
## These values taken from M1 cfg.py (https://github.com/Neurosim-lab/netpyne/blob/development/examples/M1detailed/cfg.py)
cfg.singleCellPops = False #True #False
cfg.singlePop = ''
cfg.removeWeightNorm = False

#------------------------------------------------------------------------------
# Connectivity
#------------------------------------------------------------------------------
#cfg.synWeightFractionEE = [0.5, 0.5] # E->E AMPA to NMDA ratio
#cfg.synWeightFractionEI = [0.5, 0.5] # E->I AMPA to NMDA ratio
cfg.synWeightFractionIE = dconf['syn']['synWeightFractionIE'] # [1.0] # this is set to 1 for PV,VIP; for SOM,NGF have other param for GABAASlow to GABAB ratio
cfg.synWeightFractionII = dconf['syn']['synWeightFractionII'] # [1.0]  # I->I uses single synapse mechanism

# Cortical
cfg.addConn = 1


## E/I->E/I layer weights (L1-3, L4, L5, L6)
cfg.EELayerGain = {'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0 , '5A': 1.0, '5B': 1.0, '6': 1.0}
cfg.EILayerGain = {'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0 , '5A': 1.0, '5B': 1.0, '6': 1.0}
cfg.IELayerGain = {'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0 , '5A': 1.0, '5B': 1.0, '6': 1.0}
cfg.IILayerGain = {'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0 , '5A': 1.0, '5B': 1.0, '6': 1.0}

# E -> E based on postsynaptic cortical E neuron population
# cfg.EEPopGain = {'IT2': 1.3125, 'IT3': 1.55, 'ITP4': 1.0, 'ITS4': 1.0, 'IT5A': 1.05, 'CT5A': 1.1500000000000001, 'IT5B': 0.425, 'CT5B': 1.1500000000000001, 'PT5B': 1.05, 'IT6': 1.05, 'CT6': 1.05} # this is from after generation 203 of optunaERP_23dec23_ , values used in generation 204 of the same optimization
#cfg.EEPopGain = {'IT2': 1.4125, 'IT3': 3.0, 'ITP4': 1.0, 'ITS4': 1.05, 'IT5A': 1.1500000000000001, 'CT5A': 1.6500000000000006, 'IT5B': 0.6250000000000001, 'CT5B': 3.0, 'PT5B': 3.0, 'IT6': 1.4500000000000004, 'CT6': 1.2000000000000002} # this is from after generation 59 of optunaERP_24mar5_ (optimized L3 sink)
cfg.EEPopGain = dconf['net']['EEPopGain']

# gains from E -> I based on postsynaptic cortical I neuron population
# cfg.EIPopGain = {'NGF1': 1.0, 'SOM2': 1.0, 'PV2': 1.0, 'VIP2': 1.0, 'NGF2': 1.0, 'SOM3': 1.0, 'PV3': 1.0, 'VIP3': 1.0, 'NGF3': 1.0, 'SOM4': 1.0, 'PV4': 1.0, 'VIP4': 1.0, 'NGF4': 1.0, 'SOM5A': 1.0, 'PV5A': 1.4, 'VIP5A': 1.25, 'NGF5A': 0.8, 'SOM5B': 1.0, 'PV5B': 1.4, 'VIP5B': 1.4, 'NGF5B': 0.9, 'SOM6': 1.0, 'PV6': 1.4, 'VIP6': 1.4, 'NGF6': 0.65}
# cfg.EIPopGain = {'NGF1': 1.0, 'SOM2': 1.0, 'PV2': 1.0, 'VIP2': 1.0, 'NGF2': 1.0, 'SOM3': 1.0, 'PV3': 1.0, 'VIP3': 1.0, 'NGF3': 1.0, 'SOM4': 1.0, 'PV4': 1.0, 'VIP4': 1.0, 'NGF4': 1.0, 'SOM5A': 1.0, 'PV5A': 1.4, 'VIP5A': 1.25, 'NGF5A': 0.8, 'SOM5B': 1.0, 'PV5B': 1.45, 'VIP5B': 1.4, 'NGF5B': 0.9500000000000001, 'SOM6': 1.0, 'PV6': 1.4, 'VIP6': 1.3499999999999999, 'NGF6': 0.65} # this is from after generation 203 of optunaERP_23dec23_ , values used in generation 204 of the same optimization
#cfg.EIPopGain = {'NGF1': 1.2500000000000002, 'SOM2': 0.5999999999999996, 'PV2': 0.25, 'VIP2': 1.0, 'NGF2': 1.0, 'SOM3': 3.0, 'PV3': 1.0, 'VIP3': 1.0, 'NGF3': 1.0, 'SOM4': 1.0, 'PV4': 1.1, 'VIP4': 1.0, 'NGF4': 1.0, 'SOM5A': 1.0, 'PV5A': 1.8000000000000003, 'VIP5A': 1.35, 'NGF5A': 0.9500000000000002, 'SOM5B': 1.0, 'PV5B': 3.0, 'VIP5B': 1.5, 'NGF5B': 1.4500000000000004, 'SOM6': 0.95, 'PV6': 3.0, 'VIP6': 3.0, 'NGF6': 0.25} # this is from after generation 59 of optunaERP_24mar5_ (optimized L3 sink)
cfg.EIPopGain = dconf['net']['EIPopGain']

## E->I by target cell type
cfg.EICellTypeGain = {'PV': 1.0, 'SOM': 1.0, 'VIP': 1.0, 'NGF': 1.0}

## I->E by target cell type
cfg.IECellTypeGain = {'PV': 1.0, 'SOM': 1.0, 'VIP': 1.0, 'NGF': 1.0}

# Thalamic
cfg.addIntraThalamicConn = 1.0
cfg.addCorticoThalamicConn = 1.0
cfg.addThalamoCorticalConn = 1.0

cfg.thalamoCorticalGain = 1.0
cfg.intraThalamicGain = 1.0
cfg.corticoThalamicGain = 1.0

# these params control IC -> Thalamic Core
cfg.ICThalweightECore = dconf['net']['ICThalweightECore'] # 0.8350476447841453 # 1.0218574230414905 # 1.1366391725804097  # 0.8350476447841453 # 1.0
cfg.ICThalweightICore = dconf['net']['ICThalweightICore'] # 0.2114492149101151 # 0.20065170901643178 # 0.21503725192597786 # 0.2114492149101151 # 0.25
cfg.ICThalprobECore = dconf['net']['ICThalprobECore'] # 0.163484173596043 # 0.17524000437877066 # 0.21638972066571394   # 0.163484173596043 # 0.19
cfg.ICThalprobICore = dconf['net']['ICThalprobICore'] # 0.0936669688856933 # 0.0978864963550709 # 0.11831534696879886   # 0.0936669688856933 # 0.12
# these params control IC -> Thalamic Matrix
cfg.ICThalMatrixCoreFactor = dconf['net']['ICThalMatrixCoreFactor'] # 0.1 # 0.0988423213016316 # 0.11412487872986073 # 0.1 # this is used to scale weights to thalamic matrix neurons in netParams.py
cfg.ICThalprobEMatrix = cfg.ICThalprobECore 
cfg.ICThalprobIMatrix = cfg.ICThalprobICore

# these params control cochlea -> Thalamus
cfg.cochThalweightECore = dconf['net']['cochThalweightECore'] # 0.4
cfg.cochThalweightICore = dconf['net']['cochThalweightICore'] # 0.1
cfg.cochThalprobECore = dconf['net']['cochThalprobECore'] # 0.16
cfg.cochThalprobICore = dconf['net']['cochThalprobICore'] # 0.09
cfg.cochThalMatrixCoreFactor = dconf['net']['cochThalMatrixCoreFactor'] # 0.1
cfg.cochThalprobEMatrix = cfg.cochThalprobECore
cfg.cochThalprobIMatrix = cfg.cochThalprobICore
cfg.cochThalFreqRange = dconf['net']['cochThalFreqRange'] # [1000, 2000]

# these params added from Christoph Metzner branch
# these control strength of thalamic inputs to different subpopulations
cfg.thalL4PV = dconf['net']['thalL4PV'] # 0.21367245896786016 # 0.3201033836037148 # 0.261333644625591   # 0.21367245896786016 # 0.25 
cfg.thalL4SOM = dconf['net']['thalL4SOM'] # 0.24260966747847523 # 0.3200462291706402 # 0.2612645277258505 # 0.24260966747847523 # 0.25 
cfg.thalL4E = dconf['net']['thalL4E'] # 1.9540886147587417 # 2.8510831744854714 # 2.3199103007567827   # 1.9540886147587417 # 2.0
cfg.thalL4VIP = dconf['net']['thalL4VIP'] # 1.0
cfg.thalL4NGF = dconf['net']['thalL4NGF'] # 1.0
cfg.thalL1NGF = dconf['net']['thalL1NGF'] # 1.0
cfg.ENGF1 = dconf['net']['ENGF1'] # 1.0 # modulates strength of synaptic connections from E -> NGF1 neurons

# modulates strength of connections from L4 -> L3 by different target subpopulations
#  these next 5 param values are from after generation 59 of optunaERP_24mar5_ (optimized L3 sink) 
cfg.L4L3E    = dconf['net']['L4L3E'] # 1.0188377611592279 # 1.0
cfg.L4L3PV   = dconf['net']['L4L3PV'] # 0.9829655631376849 # 1.0
cfg.L4L3SOM  = dconf['net']['L4L3SOM'] # 0.9647203483395813 # 1.0
cfg.L4L3VIP = dconf['net']['L4L3VIP'] # 1.039136847713827 # 1.0
cfg.L4L3NGF = dconf['net']['L4L3NGF'] # 0.9119964748686543 # 1.0

cfg.addSubConn = 1 # specifies to put synapses on particular subcellular target locations

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

# parameters to generate realistic  auditory thalamic inputs using Brian Hears
if dconf['sim']['useCochleaThal']:
  cfg.cochlearThalInput = dconf['CochlearThalInput']
  cti = cfg.cochlearThalInput
  cti['probECore'] = cfg.cochThalprobECore
  cti['weightECore'] = cfg.cochThalweightECore
  cti['probICore'] = cfg.cochThalprobICore
  cti['weightICore'] = cfg.cochThalweightICore
  cti['probEMatrix'] = cfg.cochThalprobEMatrix
  cti['probIMatrix'] = cfg.cochThalprobIMatrix
  cti['MatrixCoreFactor'] = cfg.cochThalMatrixCoreFactor
else:
  cfg.cochlearThalInput = False  

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

if dconf['sim']['useICThal']:
  cfg.ICThalInput = {'file': dconf['ICThalInput']['file'], # BBN_trials/ICoutput_CF_9600_10400_wav_BBN_100ms_burst_AN.mat',
                     'startTime': list(np.arange(dconf['ICThalInput']['onset'], dconf['ICThalInput']['offset'], dconf['ICThalInput']['interval'])), 
                     'weightECore': cfg.ICThalweightECore,
                     'weightICore': cfg.ICThalweightICore,
                     'probECore': cfg.ICThalprobECore, 
                     'probICore': cfg.ICThalprobICore,
                     'probEMatrix': cfg.ICThalprobEMatrix,
                     'probIMatrix': cfg.ICThalprobIMatrix,
                     'MatrixCoreFactor': cfg.ICThalMatrixCoreFactor,
                     'seed': 1}  # SHOULD THIS BE ZERO?
else:
  cfg.ICThalInput = False


cfg.scale = dconf['net']['scale'] 
cfg.sizeY = dconf['net']['sizeY'] 
cfg.sizeX = dconf['net']['sizeX'] 
cfg.sizeZ = dconf['net']['sizeZ']
cfg.scaleDensity = dconf['net']['scaleDensity'] #0.25 #1.0 #0.075 # Should be 1.0 unless need lower cell density for test simulation or visualization
  
cfg.EEGain = dconf['net']['EEGain'] 
cfg.EIGain = dconf['net']['EIGain']
cfg.IEGain = dconf['net']['IEGain']
cfg.IIGain = dconf['net']['IIGain']

cfg.EbkgThalamicGain = dconf['EbkgThalamicGain']
cfg.IbkgThalamicGain = dconf['IbkgThalamicGain']
  
"""
netParams.py 

High-level specifications for A1 network model using NetPyNE

Contributors: ericaygriffith@gmail.com, salvadordura@gmail.com, samnemo@gmail.com , & Christoph Metzner
"""

netParams = specs.NetParams()   # object of class NetParams to store the network parameters


#------------------------------------------------------------------------------
# VERSION 
#------------------------------------------------------------------------------
netParams.version = 41

#------------------------------------------------------------------------------
#
# NETWORK PARAMETERS
#
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# General network parameters
#------------------------------------------------------------------------------

netParams.scale = cfg.scale # Scale factor for number of cells # NOT DEFINED YET! 3/11/19 # How is this different than scaleDensity? 
netParams.sizeX = cfg.sizeX # x-dimension (horizontal length) size in um
netParams.sizeY = cfg.sizeY # y-dimension (vertical height or cortical depth) size in um
netParams.sizeZ = cfg.sizeZ # z-dimension (horizontal depth) size in um
netParams.shape = 'cylinder' # cylindrical (column-like) volume

#------------------------------------------------------------------------------
# General connectivity parameters
#------------------------------------------------------------------------------
netParams.scaleConnWeight = 1.0 # Connection weight scale factor (default if no model specified)
netParams.scaleConnWeightModels = { 'HH_reduced': 1.0, 'HH_full': 1.0} #scale conn weight factor for each cell model
netParams.scaleConnWeightNetStims = 1.0 #0.5  # scale conn weight factor for NetStims
netParams.defaultThreshold = 0.0 # spike threshold, 10 mV is NetCon default, lower it for all cells
netParams.defaultDelay = 2.0 # default conn delay (ms)
netParams.propVelocity = 500.0 # propagation velocity (um/ms)
netParams.probLambda = 100.0  # length constant (lambda) for connection probability decay (um)

#------------------------------------------------------------------------------
# Cell parameters
#------------------------------------------------------------------------------

Etypes = ['IT', 'ITS4', 'PT', 'CT']
Itypes = ['PV', 'SOM', 'VIP', 'NGF']
cellModels = ['HH_reduced', 'HH_full'] # List of cell models

# II: 100-950, IV: 950-1250, V: 1250-1550, VI: 1550-2000 
#layer = {'1': [0.00, 0.05], '2': [0.05, 0.08], '3': [0.08, 0.475], '4': [0.475, 0.625], '5A': [0.625, 0.667], '5B': [0.667, 0.775], '6': [0.775, 1], 'thal': [1.2, 1.4], 'cochlear': [1.6, 1.8]}  # normalized layer boundaries
layer = {'1': [0.00, 0.05], '2': [0.05, 0.08], '3': [0.08, 0.475], '4': [0.475, 0.625], '5A': [0.625, 0.667], '5B': [0.667, 0.775], '6': [0.775, 1], 'thal': [1.2, 1.4], 'cochlear': [1.6, 1.8]}  # normalized layer boundaries  

layerGroups = { '1-3': [layer['1'][0], layer['3'][1]],  # L1-3
                '4': layer['4'],                      # L4
                '5': [layer['5A'][0], layer['5B'][1]],  # L5A-5B
                '6': layer['6']}                        # L6

# add layer border correction ??
#netParams.correctBorder = {'threshold': [cfg.correctBorderThreshold, cfg.correctBorderThreshold, cfg.correctBorderThreshold], 
#                        'yborders': [layer['2'][0], layer['5A'][0], layer['6'][0], layer['6'][1]]}  # correct conn border effect


#------------------------------------------------------------------------------
## Load cell rules previously saved using netpyne format (DOES NOT INCLUDE VIP, NGF and spiny stellate)
## include conditions ('conds') for each cellRule
cellParamLabels = ['IT2_reduced', 'IT3_reduced', 'ITP4_reduced', 'ITS4_reduced',
                    'IT5A_reduced', 'CT5A_reduced', 'IT5B_reduced',
                    'PT5B_reduced', 'CT5B_reduced', 'IT6_reduced', 'CT6_reduced',
                    'PV_reduced', 'SOM_reduced', 'VIP_reduced', 'NGF_reduced',
                    'RE_reduced', 'TC_reduced', 'HTC_reduced', 'TI_reduced']  # , 'TI_reduced']

for ruleLabel in cellParamLabels:
    netParams.loadCellParamsRule(label=ruleLabel, fileName='cells/' + ruleLabel + '_cellParams.json')  # Load cellParams for each of the above cell subtype



# ## Reduce T-type calcium channel conductances (cfg.tTypeCorticalFactor ; cfg.tTypeThalamicFactor)
# for cellLabel in ['TC_reduced', 'HTC_reduced', 'RE_reduced']:
#     cellParam = netParams.cellParams[cellLabel]
#     for secName in cellParam['secs']:
#         #print('cellType: ' + cellLabel + ', section: ' + secName)
#         for mechName,mech in cellParam['secs'][secName]['mechs'].items():
#             if mechName in ['itre', 'ittc']:
#                 #print('gmax of ' + mechName + ' ' + str(cellParam['secs'][secName]['mechs'][mechName]['gmax']))  # ADD A TEST PRINT STATEMENT PRE-CHANGE
#                 cellParam['secs'][secName]['mechs'][mechName]['gmax'] *= cfg.tTypeThalamicFactor
#                 print('new gmax of ' + mechName + ' ' + str(cellParam['secs'][secName]['mechs'][mechName]['gmax']))  # ADD A TEST PRINT STATEMENT POST-CHANGE

# for cellLabel in ['TI_reduced']:
#     cellParam = netParams.cellParams[cellLabel]
#     for secName in cellParam['secs']:
#         #print('cellType: ' + cellLabel + ', section: ' + secName)
#         for mechName,mech in cellParam['secs'][secName]['mechs'].items():
#             if mechName == 'it2INT':
#                 #print('gcabar of ' + mechName + ' ' + str(cellParam['secs'][secName]['mechs'][mechName]['gcabar'])) # ADD A TEST PRINT STATEMENT PRE-CHANGE
#                 cellParam['secs'][secName]['mechs'][mechName]['gcabar'] *= cfg.tTypeThalamicFactor
#                 print('new gcabar of ' + mechName + ' ' + str(cellParam['secs'][secName]['mechs'][mechName]['gcabar'])) # ADD A TEST PRINT STATEMENT POST-CHANGE

# for cellLabel in ['IT2_reduced', 'IT3_reduced', 'ITP4_reduced', 'ITS4_reduced',
#                     'IT5A_reduced', 'CT5A_reduced', 'IT5B_reduced', 'CT5B_reduced', 
#                     'IT6_reduced', 'CT6_reduced']:
#     cellParam = netParams.cellParams[cellLabel]
#     for secName in cellParam['secs']:
#         #print('cellType: ' + cellLabel + ', section: ' + secName)
#         for mechName,mech in cellParam['secs'][secName]['mechs'].items():
#             if mechName == 'cat':
#                 #print('gcatbar of ' + mechName + ' ' + str(cellParam['secs'][secName]['mechs'][mechName]['gcatbar']))# ADD A TEST PRINT STATEMENT PRE-CHANGE
#                 cellParam['secs'][secName]['mechs'][mechName]['gcatbar'] *= cfg.tTypeCorticalFactor
#                 print('new gcatbar of ' + mechName + ' ' + str(cellParam['secs'][secName]['mechs'][mechName]['gcatbar'])) # ADD A TEST PRINT STATEMENT POST-CHANGE



## Manipulate NMDAR weights to Inhibitory Populations 

# # Thalamic Interneuron Version:
# TI_version = 'default' # IAHP # IL # default

# if TI_version == 'IAHP': 
#     netParams.loadCellParamsRule(label='TI_reduced', fileName='cells/TI_reduced_cellParams_IAHP.json') 
#     print('IAHP reduced conductance version loaded')
# elif TI_version == 'IL':
#     netParams.loadCellParamsRule(label='TI_reduced', fileName='cells/TI_reduced_cellParams_IL.json') 
#     print('IL reduced conductance version loaded')
# else: 
#     netParams.loadCellParamsRule(label='TI_reduced', fileName='cells/TI_reduced_cellParams.json')
#     print('Default thal int model loaded')



# # change weightNorm 
# for k in cfg.weightNormScaling:
#     for sec in netParams.cellParams[k]['secs'].values():
#         for i in range(len(sec['weightNorm'])):
#             sec['weightNorm'][i] *= cfg.weightNormScaling[k]


# # Parametrize PT ih_gbar and exc cells K_gmax to simulate NA/ACh neuromodulation
# for cellLabel in ['PT5B_reduced']:
#     cellParam = netParams.cellParams[cellLabel] 

#     for secName in cellParam['secs']:
#         # Adapt ih params based on cfg param
#         for mechName,mech in cellParam['secs'][secName]['mechs'].items():
#             if mechName in ['ih','h','h15', 'hd']: 
#                 mech['gbar'] = [g*cfg.ihGbar for g in mech['gbar']] if isinstance(mech['gbar'],list) else mech['gbar']*cfg.ihGbar


# # Adapt Kgbar
# for cellLabel in ['IT2_reduced', 'IT3_reduced', 'ITP4_reduced', 'ITS4_reduced',
#                     'IT5A_reduced', 'CT5A_reduced', 'IT5B_reduced',
#                     'PT5B_reduced', 'CT5B_reduced', 'IT6_reduced', 'CT6_reduced']:
#     cellParam = netParams.cellParams[cellLabel] 

#     for secName in cellParam['secs']:
#         for kmech in [k for k in cellParam['secs'][secName]['mechs'].keys() if k in ['kap','kdr']]:
#             cellParam['secs'][secName]['mechs'][kmech]['gbar'] *= cfg.KgbarFactor 

#------------------------------------------------------------------------------
# Population parameters
#------------------------------------------------------------------------------

## load densities
with open('cells/cellDensity.pkl', 'rb') as fileObj: density = pickle.load(fileObj)['density']
density = {k: [x * cfg.scaleDensity for x in v] for k,v in density.items()} # Scale densities 

# ### LAYER 1:
netParams.popParams['NGF1'] = {'cellType': 'NGF', 'cellModel': 'HH_reduced','ynormRange': layer['1'],   'density': density[('A1','nonVIP')][0]}

### LAYER 2:
netParams.popParams['IT2'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': layer['2'],   'density': density[('A1','E')][1]}     # cfg.cellmod for 'cellModel' in M1 netParams.py 
netParams.popParams['SOM2'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',   'ynormRange': layer['2'],   'density': density[('A1','SOM')][1]}   
netParams.popParams['PV2'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',   'ynormRange': layer['2'],   'density': density[('A1','PV')][1]}    
netParams.popParams['VIP2'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',   'ynormRange': layer['2'],   'density': density[('A1','VIP')][1]}
netParams.popParams['NGF2'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',   'ynormRange': layer['2'],   'density': density[('A1','nonVIP')][1]}

### LAYER 3:
netParams.popParams['IT3'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': layer['3'],   'density': density[('A1','E')][1]} ## CHANGE DENSITY
netParams.popParams['SOM3'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',   'ynormRange': layer['3'],   'density': density[('A1','SOM')][1]} ## CHANGE DENSITY
netParams.popParams['PV3'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',   'ynormRange': layer['3'],   'density': density[('A1','PV')][1]} ## CHANGE DENSITY
netParams.popParams['VIP3'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',   'ynormRange': layer['3'],   'density': density[('A1','VIP')][1]} ## CHANGE DENSITY
netParams.popParams['NGF3'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',   'ynormRange': layer['3'],   'density': density[('A1','nonVIP')][1]}


### LAYER 4: 
netParams.popParams['ITP4'] =	 {'cellType': 'IT', 'cellModel': 'HH_reduced',  'ynormRange': layer['4'],   'density': 0.5*density[('A1','E')][2]}      ## CHANGE DENSITY #
netParams.popParams['ITS4'] =	 {'cellType': 'IT', 'cellModel': 'HH_reduced', 'ynormRange': layer['4'],  'density': 0.5*density[('A1','E')][2]}      ## CHANGE DENSITY 
netParams.popParams['SOM4'] = 	 {'cellType': 'SOM', 'cellModel': 'HH_reduced',   'ynormRange': layer['4'],  'density': density[('A1','SOM')][2]}
netParams.popParams['PV4'] = 	 {'cellType': 'PV', 'cellModel': 'HH_reduced',   'ynormRange': layer['4'],   'density': density[('A1','PV')][2]}
netParams.popParams['VIP4'] =	 {'cellType': 'VIP', 'cellModel': 'HH_reduced',   'ynormRange': layer['4'],  'density': density[('A1','VIP')][2]}
netParams.popParams['NGF4'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',   'ynormRange': layer['4'],  'density': density[('A1','nonVIP')][2]}

# # ### LAYER 5A: 
netParams.popParams['IT5A'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5A'], 	'density': 0.5*density[('A1','E')][3]}      
netParams.popParams['CT5A'] =     {'cellType': 'CT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5A'],   'density': 0.5*density[('A1','E')][3]}  # density is [5] because we are using same numbers for L5A and L6 for CT cells? 
netParams.popParams['SOM5A'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',    'ynormRange': layer['5A'],	'density': density[('A1','SOM')][3]}          
netParams.popParams['PV5A'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',    'ynormRange': layer['5A'],	'density': density[('A1','PV')][3]}         
netParams.popParams['VIP5A'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',    'ynormRange': layer['5A'],   'density': density[('A1','VIP')][3]}
netParams.popParams['NGF5A'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',    'ynormRange': layer['5A'],   'density': density[('A1','nonVIP')][3]}

### LAYER 5B: 
netParams.popParams['IT5B'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5B'], 	'density': (1/3)*density[('A1','E')][4]}  
netParams.popParams['CT5B'] =     {'cellType': 'CT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5B'],   'density': (1/3)*density[('A1','E')][4]}  # density is [5] because we are using same numbers for L5B and L6 for CT cells? 
netParams.popParams['PT5B'] =     {'cellType': 'PT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5B'], 	'density': (1/3)*density[('A1','E')][4]}  
netParams.popParams['SOM5B'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',    'ynormRange': layer['5B'],   'density': density[('A1', 'SOM')][4]}
netParams.popParams['PV5B'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',    'ynormRange': layer['5B'],	'density': density[('A1','PV')][4]}     
netParams.popParams['VIP5B'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',    'ynormRange': layer['5B'],   'density': density[('A1','VIP')][4]}
netParams.popParams['NGF5B'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',    'ynormRange': layer['5B'],   'density': density[('A1','nonVIP')][4]}

# # ### LAYER 6:
netParams.popParams['IT6'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': layer['6'],   'density': 0.5*density[('A1','E')][5]}  
netParams.popParams['CT6'] =     {'cellType': 'CT',  'cellModel': 'HH_reduced',  'ynormRange': layer['6'],   'density': 0.5*density[('A1','E')][5]} 
netParams.popParams['SOM6'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',   'ynormRange': layer['6'],   'density': density[('A1','SOM')][5]}   
netParams.popParams['PV6'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',   'ynormRange': layer['6'],   'density': density[('A1','PV')][5]}     
netParams.popParams['VIP6'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',   'ynormRange': layer['6'],   'density': density[('A1','VIP')][5]}
netParams.popParams['NGF6'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',   'ynormRange': layer['6'],   'density': density[('A1','nonVIP')][5]}


### THALAMIC POPULATIONS (from prev model)
thalDensity = density[('A1','PV')][2] * 1.25  # temporary estimate (from prev model)

netParams.popParams['TC'] =     {'cellType': 'TC',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': 0.75*thalDensity}  
netParams.popParams['TCM'] =    {'cellType': 'TC',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': thalDensity} 
netParams.popParams['HTC'] =    {'cellType': 'HTC', 'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': 0.25*thalDensity}   
netParams.popParams['IRE'] =    {'cellType': 'RE',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': thalDensity}     
netParams.popParams['IREM'] =   {'cellType': 'RE', 'cellModel': 'HH_reduced',   'ynormRange': layer['thal'],   'density': thalDensity}
netParams.popParams['TI'] =     {'cellType': 'TI',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': 0.33 * thalDensity} ## Winer & Larue 1996; Huang et al 1999 
netParams.popParams['TIM'] =    {'cellType': 'TI',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': 0.33 * thalDensity} ## Winer & Larue 1996; Huang et al 1999 


if cfg.singleCellPops:
    for pop in netParams.popParams.values(): pop['numCells'] = 1

## List of E and I pops to use later on
Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'CT5B' , 'PT5B', 'IT6', 'CT6']  # all layers

Ipops = ['NGF1',                            # L1
        'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
        'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
        'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
        'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A  
        'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
        'PV6', 'SOM6', 'VIP6', 'NGF6']      # L6 



#------------------------------------------------------------------------------
# Synaptic mechanism parameters
#------------------------------------------------------------------------------

### From M1 detailed netParams.py 
netParams.synMechParams['NMDA'] = {'mod': 'MyExp2SynNMDABB', 'tau1NMDA': 15, 'tau2NMDA': 150, 'e': 0}
netParams.synMechParams['AMPA'] = {'mod':'MyExp2SynBB', 'tau1': 0.05, 'tau2': 5.3*cfg.AMPATau2Factor, 'e': 0}
netParams.synMechParams['GABAB'] = dconf['syn']['GABAB'] # {'mod':'MyExp2SynBB', 'tau1': 3.5, 'tau2': 260.9, 'e': -93} 
netParams.synMechParams['GABAA'] = {'mod':'MyExp2SynBB', 'tau1': 0.07, 'tau2': 18.2, 'e': -80}
netParams.synMechParams['GABAA_VIP'] = {'mod':'MyExp2SynBB', 'tau1': 0.3, 'tau2': 6.4, 'e': -80}  # Pi et al 2013
netParams.synMechParams['GABAASlow'] = {'mod': 'MyExp2SynBB','tau1': 2, 'tau2': 100, 'e': -80}
netParams.synMechParams['GABAASlowSlow'] = {'mod': 'MyExp2SynBB', 'tau1': 200, 'tau2': 400, 'e': -80}

ESynMech = ['AMPA', 'NMDA']
SOMESynMech = ['GABAASlow','GABAB']
SOMISynMech = ['GABAASlow']
PVSynMech = ['GABAA']
VIPSynMech = ['GABAA_VIP']
NGFESynMech = ['GABAA', 'GABAB']
NGFISynMech = ['GABAA']
ThalIESynMech = ['GABAASlow','GABAB']
ThalIISynMech = ['GABAASlow']

#------------------------------------------------------------------------------
# Local connectivity parameters
#------------------------------------------------------------------------------

## load data from conn pre-processing file
with open('conn/conn.pkl', 'rb') as fileObj: connData = pickle.load(fileObj)
pmat = connData['pmat']
lmat = connData['lmat']
wmat = connData['wmat']
bins = connData['bins']
connDataSource = connData['connDataSource']

wmat = cfg.wmat

def wireCortex ():
  layerGainLabels = ['1', '2', '3', '4', '5A', '5B', '6']
  #------------------------------------------------------------------------------
  ## E -> E
  if cfg.EEGain > 0.0:
      for pre in Epops:
          for post in Epops:
              for l in layerGainLabels:  # used to tune each layer group independently
                  scaleFactor = 1.0
                  if connDataSource['E->E/I'] in ['Allen_V1', 'Allen_custom']:
                      prob = '%f * exp(-dist_2D/%f)' % (pmat[pre][post], lmat[pre][post])
                  else:
                      prob = pmat[pre][post]
                  if pre=='ITS4' or pre=='ITP4':
                      if post=='IT3':
                          scaleFactor = cfg.L4L3E#25
                  netParams.connParams['EE_'+pre+'_'+post+'_'+l] = { 
                      'preConds': {'pop': pre}, 
                      'postConds': {'pop': post, 'ynorm': layer[l]},
                      'synMech': ESynMech,
                      'probability': prob,
                      'weight': wmat[pre][post] * cfg.EEGain * cfg.EELayerGain[l] * cfg.EEPopGain[post] * scaleFactor, 
                      'synMechWeightFactor': cfg.synWeightFractionEE,
                      'delay': 'defaultDelay+dist_3D/propVelocity',
                      'synsPerConn': 1,
                      'sec': 'dend_all'}                    
  #------------------------------------------------------------------------------
  ## E -> I       ## MODIFIED FOR NMDAR MANIPULATION!! 
  if cfg.EIGain > 0.0:
      for pre in Epops:
          for post in Ipops:
              for postType in Itypes:
                  if postType in post: # only create rule if celltype matches pop
                      for l in layerGainLabels:  # used to tune each layer group independently
                          scaleFactor = 1.0
                          if connDataSource['E->E/I'] in ['Allen_V1', 'Allen_custom']:
                              prob = '%f * exp(-dist_2D/%f)' % (pmat[pre][post], lmat[pre][post])
                          else:
                              prob = pmat[pre][post]                        
                          if 'NGF' in post:
                              synWeightFactor = cfg.synWeightFractionENGF   
                          elif 'PV' in post:
                              synWeightFactor = cfg.synWeightFractionEI_CustomCort
                          else:
                              synWeightFactor = cfg.synWeightFractionEI #cfg.synWeightFractionEI_CustomCort  #cfg.synWeightFractionEI   
                          if 'NGF1' in post:
                              scaleFactor = cfg.ENGF1  
                          if pre=='ITS4' or pre=='ITP4':
                              if post=='PV3':
                                  scaleFactor = cfg.L4L3PV#25
                              elif post=='SOM3':
                                  scaleFactor = cfg.L4L3SOM
                              elif post=='NGF3':
                                  scaleFactor = cfg.L4L3NGF#25
                              elif post=='VIP3':
                                  scaleFactor = cfg.L4L3VIP#25
                          netParams.connParams['EI_'+pre+'_'+post+'_'+postType+'_'+l] = { 
                              'preConds': {'pop': pre}, 
                              'postConds': {'pop': post, 'cellType': postType, 'ynorm': layer[l]},
                              'synMech': ESynMech,
                              'probability': prob,
                              'weight': wmat[pre][post] * cfg.EIGain * cfg.EICellTypeGain[postType] * cfg.EILayerGain[l] * cfg.EIPopGain[post] * scaleFactor, 
                              'synMechWeightFactor': synWeightFactor,
                              'delay': 'defaultDelay+dist_3D/propVelocity',
                              'synsPerConn': 1,
                              'sec': 'proximal'}                
  # cfg.NMDARfactor * wmat[pre][post] * cfg.EIGain * cfg.EICellTypeGain[postType] * cfg.EILayerGain[l]]
  #------------------------------------------------------------------------------
  ## I -> E
  if cfg.IEGain > 0.0:
      if connDataSource['I->E/I'] == 'Allen_custom':
          for pre in Ipops:
              for preType in Itypes:
                  if preType in pre:  # only create rule if celltype matches pop
                      for post in Epops:
                          for l in layerGainLabels:  # used to tune each layer group independently                            
                              prob = '%f * exp(-dist_2D/%f)' % (pmat[pre][post], lmat[pre][post])
                              synWeightFactor = cfg.synWeightFractionIE
                              if 'SOM' in pre:
                                  synMech = SOMESynMech
                                  synWeightFactor = cfg.synWeightFractionSOM['E']
                              elif 'PV' in pre:
                                  synMech = PVSynMech
                              elif 'VIP' in pre:
                                  synMech = VIPSynMech
                              elif 'NGF' in pre:
                                  synMech = NGFESynMech
                                  synWeightFactor = cfg.synWeightFractionNGF['E']
                              netParams.connParams['IE_'+pre+'_'+preType+'_'+post+'_'+l] = { 
                                  'preConds': {'pop': pre}, 
                                  'postConds': {'pop': post, 'ynorm': layer[l]},
                                  'synMech': synMech,
                                  'probability': prob,
                                  'weight': wmat[pre][post] * cfg.IEGain * cfg.IECellTypeGain[preType] * cfg.IELayerGain[l], 
                                  'synMechWeightFactor': synWeightFactor,
                                  'delay': 'defaultDelay+dist_3D/propVelocity',
                                  'synsPerConn': 1,
                                  'sec': 'proximal'}                    
  #------------------------------------------------------------------------------
  ## I -> I
  if cfg.IIGain > 0.0:
      if connDataSource['I->E/I'] == 'Allen_custom':
          for pre in Ipops:
              for post in Ipops:
                  for l in layerGainLabels:                     
                      prob = '%f * exp(-dist_2D/%f)' % (pmat[pre][post], lmat[pre][post])
                      synWeightFactor = cfg.synWeightFractionII
                      if 'SOM' in pre:
                          synMech = SOMISynMech
                          synWeightFactor = cfg.synWeightFractionSOM['I']
                      elif 'PV' in pre:
                          synMech = PVSynMech
                      elif 'VIP' in pre:
                          synMech = VIPSynMech
                      elif 'NGF' in pre:
                          synMech = NGFISynMech
                          synWeightFactor = cfg.synWeightFractionNGF['I']                          
                      netParams.connParams['II_'+pre+'_'+post+'_'+l] = { 
                          'preConds': {'pop': pre}, 
                          'postConds': {'pop': post,  'ynorm': layer[l]},
                          'synMech': synMech,
                          'probability': prob,
                          'weight': wmat[pre][post] * cfg.IIGain * cfg.IILayerGain[l], 
                          'synMechWeightFactor': synWeightFactor,
                          'delay': 'defaultDelay+dist_3D/propVelocity',
                          'synsPerConn': 1,
                          'sec': 'proximal'}                        

if cfg.addConn: wireCortex()                      
                    
#------------------------------------------------------------------------------
# Thalamic connectivity parameters
#------------------------------------------------------------------------------
TEpops = ['TC', 'TCM', 'HTC']
TIpops = ['IRE', 'IREM', 'TI', 'TIM']

def IsThalamicCore (ct): return ct == 'TC' or ct == 'HTC' or ct == 'IRE' or ct == 'TI'

def wireThal ():
  # set intrathalamic connections
  for pre in TEpops+TIpops:
      for post in TEpops+TIpops:
          if post in pmat[pre]:
              # for syns use ESynMech, ThalIESynMech and ThalIISynMech
              if pre in TEpops:     # E->E/I
                  syn = ESynMech
                  synWeightFactor = cfg.synWeightFractionEE
              elif post in TEpops:  # I->E
                  syn = ThalIESynMech
                  synWeightFactor = dconf['syn']['synWeightFractionThal']['Thal']['I']['E'] 
              else:                  # I->I
                  syn = ThalIISynMech
                  synWeightFactor = dconf['syn']['synWeightFractionThal']['Thal']['I']['I'] 
              # use spatially dependent wiring between thalamic core excitatory neurons
              if (pre == 'TC' and (post == 'TC' or post == 'HTC')) or (pre == 'HTC' and (post == 'TC' or post == 'HTC')):
                prob = '%f * exp(-dist_x/%f)' % (pmat[pre][post], dconf['net']['ThalamicCoreLambda'])
              else:
                prob = pmat[pre][post]  
              netParams.connParams['ITh_'+pre+'_'+post] = { 
                  'preConds': {'pop': pre}, 
                  'postConds': {'pop': post},
                  'synMech': syn,
                  'probability': prob,
                  'weight': wmat[pre][post] * cfg.intraThalamicGain, 
                  'synMechWeightFactor': synWeightFactor,
                  'delay': 'defaultDelay+dist_3D/propVelocity',
                  'synsPerConn': 1,
                  'sec': 'soma'}

if cfg.addConn and cfg.addIntraThalamicConn: wireThal()              

#-----------------------------------------------------------------------------
def connectCortexToThal ():
  # corticothalamic connections
  for pre in Epops:
      for post in TEpops+TIpops:
          if post in pmat[pre]:
              if IsThalamicCore(post): # use spatially dependent wiring for thalamic core
                prob = '%f * exp(-dist_x/%f)' % (pmat[pre][post], dconf['net']['ThalamicCoreLambda'])
              else:
                prob = pmat[pre][post]              
              netParams.connParams['CxTh_'+pre+'_'+post] = { 
                  'preConds': {'pop': pre}, 
                  'postConds': {'pop': post},
                  'synMech': ESynMech,
                  'probability': prob,
                  'weight': wmat[pre][post] * cfg.corticoThalamicGain, 
                  'synMechWeightFactor': cfg.synWeightFractionEE,
                  'delay': 'defaultDelay+dist_3D/propVelocity',
                  'synsPerConn': 1,
                  'sec': 'soma'}

if cfg.addConn and cfg.addCorticoThalamicConn: connectCortexToThal()              

#------------------------------------------------------------------------------
def connectThalToCortex ():
  # thalamocortical connections, some params added from Christoph Metzner's branch
  for pre in TEpops+TIpops:
      for post in Epops+Ipops:
          scaleFactor = 1.0
          if post in pmat[pre]:
              if IsThalamicCore(pre): # use spatially dependent wiring for thalamic core
                prob = '%f * exp(-dist_x/%f)' % (pmat[pre][post], dconf['net']['ThalamicCoreLambda']) # NB: should check if this is ok 
              else:
                prob = '%f * exp(-dist_2D/%f)' % (pmat[pre][post], lmat[pre][post]) # NB: check what the 2D inverse distance based on. lmat from conn/conn.pkl
              # for syns use ESynMech, SOMESynMech and SOMISynMech 
              if pre in TEpops:     # E->E/I
                  if post=='PV4':
                      syn = ESynMech
                      synWeightFactor = cfg.synWeightFractionEE
                      scaleFactor = cfg.thalL4PV#25
                  elif post=='SOM4':
                      syn = ESynMech
                      synWeightFactor = cfg.synWeightFractionEE
                      scaleFactor = cfg.thalL4SOM
                  elif post=='ITS4':
                      syn = ESynMech
                      synWeightFactor = cfg.synWeightFractionEE
                      scaleFactor = cfg.thalL4E#25
                  elif post=='ITP4':
                      syn = ESynMech
                      synWeightFactor = cfg.synWeightFractionEE
                      scaleFactor = cfg.thalL4E#25
                  elif post=='NGF4':
                      syn = ESynMech
                      synWeightFactor = cfg.synWeightFractionEE
                      scaleFactor = cfg.thalL4NGF#25
                  elif post=='VIP4':
                      syn = ESynMech
                      synWeightFactor = cfg.synWeightFractionEE
                      scaleFactor = cfg.thalL4VIP#25
                  elif post=='NGF1':
                      syn = ESynMech
                      synWeightFactor = cfg.synWeightFractionEE
                      scaleFactor = cfg.thalL1NGF#25
                  else:
                      syn = ESynMech
                      synWeightFactor = cfg.synWeightFractionEE
              elif post in Epops:  # I->E
                  syn = ThalIESynMech
                  synWeightFactor = cfg.synWeightFractionThal['Ctx']['I']['E']
              else:                  # I->I
                  syn = ThalIISynMech
                  synWeightFactor = fg.synWeightFractionThal['Ctx']['I']['I']
              netParams.connParams['ThCx_'+pre+'_'+post] = { 
                  'preConds': {'pop': pre}, 
                  'postConds': {'pop': post},
                  'synMech': syn,
                  'probability': prob,
                  'weight': wmat[pre][post] * cfg.thalamoCorticalGain * scaleFactor, 
                  'synMechWeightFactor': synWeightFactor,
                  'delay': 'defaultDelay+dist_3D/propVelocity',
                  'synsPerConn': 1,
                  'sec': 'soma'}                  

if cfg.addConn and cfg.addThalamoCorticalConn: connectThalToCortex()
              
#------------------------------------------------------------------------------
# Subcellular connectivity (synaptic distributions)
#------------------------------------------------------------------------------  
# Set target sections (somatodendritic distribution of synapses)
# From Billeh 2019 (Allen V1) (fig 4F) and Tremblay 2016 (fig 3)
def addSubConn ():
  #------------------------------------------------------------------------------
  # E -> E2/3,4: soma,dendrites <200um
  netParams.subConnParams['E->E2,3,4'] = {
      'preConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']}, 
      'postConds': {'pops': ['IT2', 'IT3', 'ITP4', 'ITS4']},
      'sec': 'proximal',
      'groupSynMechs': ESynMech, 
      'density': 'uniform'} 
  #------------------------------------------------------------------------------
  # E -> E5,6: soma,dendrites (all)
  netParams.subConnParams['E->E5,6'] = {
      'preConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']}, 
      'postConds': {'pops': ['IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6']},
      'sec': 'all',
      'groupSynMechs': ESynMech, 
      'density': 'uniform'}
  #------------------------------------------------------------------------------
  # E -> I: soma, dendrite (all)
  netParams.subConnParams['E->I'] = {
      'preConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']}, 
      'postConds': {'cellType': ['PV','SOM','NGF', 'VIP']},
      'sec': 'all',
      'groupSynMechs': ESynMech, 
      'density': 'uniform'} 
  #------------------------------------------------------------------------------
  # NGF1 -> E: apic_tuft
  netParams.subConnParams['NGF1->E'] = {
      'preConds': {'pops': ['NGF1']}, 
      'postConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']},
      'sec': 'apic_tuft',
      'groupSynMechs': NGFESynMech, 
      'density': 'uniform'} 
  #------------------------------------------------------------------------------
  # NGF2,3,4 -> E2,3,4: apic_trunk
  netParams.subConnParams['NGF2,3,4->E2,3,4'] = {
      'preConds': {'pops': ['NGF2', 'NGF3', 'NGF4']}, 
      'postConds': {'pops': ['IT2', 'IT3', 'ITP4', 'ITS4']},
      'sec': 'apic_trunk',
      'groupSynMechs': NGFESynMech, 
      'density': 'uniform'} 
  #------------------------------------------------------------------------------
  # NGF2,3,4 -> E5,6: apic_uppertrunk
  netParams.subConnParams['NGF2,3,4->E5,6'] = {
      'preConds': {'pops': ['NGF2', 'NGF3', 'NGF4']}, 
      'postConds': {'pops': ['IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6']},
      'sec': 'apic_uppertrunk',
      'groupSynMechs': NGFESynMech, 
      'density': 'uniform'} 
  #------------------------------------------------------------------------------
  # NGF5,6 -> E5,6: apic_lowerrunk
  netParams.subConnParams['NGF5,6->E5,6'] = {
      'preConds': {'pops': ['NGF5A', 'NGF5B', 'NGF6']}, 
      'postConds': {'pops': ['IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6']},
      'sec': 'apic_lowertrunk',
      'groupSynMechs': NGFESynMech, 
      'density': 'uniform'} 
  #------------------------------------------------------------------------------
  #  SOM -> E: all_dend (not close to soma)
  netParams.subConnParams['SOM->E'] = {
      'preConds': {'cellType': ['SOM']}, 
      'postConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']},
      'sec': 'dend_all',
      'groupSynMechs': SOMESynMech, 
      'density': 'uniform'} 
  #------------------------------------------------------------------------------
  #  PV -> E: proximal
  netParams.subConnParams['PV->E'] = {
      'preConds': {'cellType': ['PV']}, 
      'postConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']},
      'sec': 'proximal',
      'groupSynMechs': PVSynMech, 
      'density': 'uniform'} 
  #------------------------------------------------------------------------------
  #  TC -> E: proximal
  netParams.subConnParams['TC->E'] = {
      'preConds': {'cellType': ['TC', 'HTC']}, 
      'postConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']},
      'sec': 'proximal',
      'groupSynMechs': ESynMech, 
      'density': 'uniform'} 
  #------------------------------------------------------------------------------
  #  TCM -> E: apical
  netParams.subConnParams['TCM->E'] = {
      'preConds': {'cellType': ['TCM']}, 
      'postConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']},
      'sec': 'apic',
      'groupSynMechs': ESynMech, 
      'density': 'uniform'}

if cfg.addSubConn: addSubConn()  

#------------------------------------------------------------------------------
# Background inputs 
#------------------------------------------------------------------------------  
if cfg.addBkgConn:
    # add bkg sources for E and I cells
    netParams.stimSourceParams['excBkg'] = {'type': 'NetStim', 'start': cfg.startBkg, 'rate': cfg.rateBkg['exc'], 'noise': cfg.noiseBkg, 'number': 1e9}
    netParams.stimSourceParams['inhBkg'] = {'type': 'NetStim', 'start': cfg.startBkg, 'rate': cfg.rateBkg['inh'], 'noise': cfg.noiseBkg, 'number': 1e9}
    
    if cfg.cochlearThalInput:
        from input import cochlearSpikes
        dcoch = cochlearSpikes(freqRange = cfg.cochlearThalInput['freqRange'],
                               numCenterFreqs=cfg.cochlearThalInput['numCenterFreqs'],
                               loudnessScale=cfg.cochlearThalInput['loudnessScale'],
                               lfnwave=cfg.cochlearThalInput['lfnwave'],
                               lonset=cfg.cochlearThalInput['lonset'])
        cochlearSpkTimes = dcoch['spkT']
        cfg.cochlearCenterFreqs = cochlearCenterFreqs = dcoch['cf']
        cfg.numCochlearCells = numCochlearCells = len(cochlearCenterFreqs)
        netParams.popParams['cochlea'] = {'cellModel': 'VecStim', 'numCells': numCochlearCells, 'spkTimes': cochlearSpkTimes, 'ynormRange': layer['cochlear']}
        # netParams.popParams['cochlea'] = {'cellModel': 'VecStim', 'numCells': numCochlearCells}#, 'spkTimes': cochlearSpkTimes, 'ynormRange': layer['cochlear']}
        # netParams.popParams['cochlea'] = {'cellModel': 'NetStim', 'numCells': 1000, 'interval': 100}#, 'spkTimes': cochlearSpkTimes, 'ynormRange': layer['cochlear']}

        # netParams.popParams['jnk'] = {'cellModel': 'NetStim', 'numCells': 1000, 'interval': 100}#, 'spkTimes': cochlearSpkTimes, 'ynormRange': layer['cochlear']}
        # netParams.popParams['funk'] = {'cellModel': 'IntFire1', 'numCells': 1000}#, 'spkTimes': cochlearSpkTimes, 'ynormRange': layer['cochlear']}                
        # print('len(cochlearSpkTimes):',len(cochlearSpkTimes))
        # netParams.popParams['cochlea']['gridSpacing'] = 1
        # netParams.popParams['cochlea']['sizeX'] = 100000 # numCochlearCells + 1
        # netParams.popParams['cochlea']['sizeY'] = netParams.popParams['cochlea']['sizeZ'] = 1

    if cfg.ICThalInput:
        # load file with IC output rates
        from scipy.io import loadmat
        import numpy as np

        data = loadmat(cfg.ICThalInput['file'])
        fs = data['RsFs'][0][0]
        ICrates = data['BE_sout_population'].tolist()
        ICrates = [x + [0] for x in ICrates] # add trailing zero to avoid long output from inh_poisson_generator() 
        ICtimes = list(np.arange(0, cfg.duration, 1000./fs))  # list with times to set each time-dep rate
        
        ICrates = ICrates * 4 # 200 cells
        
        numCells = len(ICrates)

        # these next two parameters are derived, so should be set here in case used by batch/optimization; because cfg.py
        # gets copied as a .json file without re-interpreting the other variables
        cfg.ICThalInput['weightEMatrix'] = cfg.ICThalInput['weightECore'] * cfg.ICThalInput['MatrixCoreFactor']
        cfg.ICThalInput['weightIMatrix'] = cfg.ICThalInput['weightICore'] * cfg.ICThalInput['MatrixCoreFactor']        

        # Option 1: create population of DynamicNetStims with time-varying rates
        #netParams.popParams['IC'] = {'cellModel': 'DynamicNetStim', 'numCells': numCells, 'ynormRange': layer['cochlear'],
        #    'dynamicRates': {'rates': ICrates, 'times': ICtimes}}

        # Option 2:
        from input import inh_poisson_generator
        
        maxLen = min(len(ICrates[0]), len(ICtimes))

        ### ADDED BY EYG 9/23/2022 TO ALLOW FOR MULTIPLE START TIMES (TEST)
        if type(cfg.ICThalInput['startTime']) == list:

            from collections import OrderedDict
            spkTimesDict = OrderedDict()
            startTimes = cfg.ICThalInput['startTime']
            for startTime in startTimes:
                keyName = 'startTime_' + str(startTime)
                print('KEY NAME: ' + keyName)
                spkTimesDict[keyName] = [[x+startTime for x in inh_poisson_generator(ICrates[i][:maxLen], ICtimes[:maxLen], cfg.duration, cfg.ICThalInput['seed']+i)] for i in range(len(ICrates))]

            spkTimes = [None] * len(ICrates) #[]

            for i in range(len(ICrates)):
                for key in spkTimesDict.keys():
                    if spkTimes[i] is None:
                        spkTimes[i] = spkTimesDict[key][i]
                    else:
                        spkTimes[i] = spkTimes[i] + spkTimesDict[key][i]

        else:
            ## Can change the t_stop arg in the inh_poisson_generator fx from cfg.duration to the length of the BBN stimulus (>100 ms)
            spkTimes = [[x+cfg.ICThalInput['startTime'] for x in inh_poisson_generator(ICrates[i][:maxLen], ICtimes[:maxLen], cfg.duration, cfg.ICThalInput['seed']+i)] for i in range(len(ICrates))]

        netParams.popParams['IC'] = {'cellModel': 'VecStim', 'numCells': numCells, 'ynormRange': layer['cochlear'],
            'spkTimes': spkTimes}

    # excBkg/I -> thalamus + cortex
    with open('cells/bkgWeightPops.json', 'r') as f:
        weightBkg = json.load(f)
    pops = list(cfg.allpops)
    pops.remove('IC')
    pops.remove('cochlea')
    for pop in ['TC', 'TCM', 'HTC']:
        weightBkg[pop] *= cfg.EbkgThalamicGain 
    for pop in ['IRE', 'IREM', 'TI', 'TIM']:
        weightBkg[pop] *= cfg.IbkgThalamicGain 
    for pop in pops:
        netParams.stimTargetParams['excBkg->'+pop] =  {
            'source': 'excBkg', 
            'conds': {'pop': pop},
            'sec': 'apic', 
            'loc': 0.5,
            'synMech': ESynMech,
            'weight': weightBkg[pop],
            'synMechWeightFactor': cfg.synWeightFractionEE,
            'delay': cfg.delayBkg}
        netParams.stimTargetParams['inhBkg->'+pop] =  {
            'source': 'inhBkg', 
            'conds': {'pop': pop},
            'sec': 'proximal',
            'loc': 0.5,
            'synMech': 'GABAA',
            'weight': weightBkg[pop],
            'delay': cfg.delayBkg}
        
    # cochlea/IC -> thal
    if cfg.ICThalInput:
        # IC -> thalamic core
        netParams.connParams['IC->ThalECore'] = { 
            'preConds': {'pop': 'IC'}, 
            'postConds': {'pop': ['TC', 'HTC']},
            'sec': 'soma', 
            'loc': 0.5,
            'synMech': ESynMech,
            'probability': cfg.ICThalInput['probECore'],
            'weight': cfg.ICThalInput['weightECore'],
            'synMechWeightFactor': cfg.synWeightFractionEE,
            'delay': cfg.delayBkg}        
        netParams.connParams['IC->ThalICore'] = { 
            'preConds': {'pop': 'IC'}, 
            'postConds': {'pop': ['IRE', 'TI']},
            'sec': 'soma', 
            'loc': 0.5,
            'synMech': 'GABAA',
            'probability': cfg.ICThalInput['probICore'],
            'weight': cfg.ICThalInput['weightICore'],
            'delay': cfg.delayBkg}
        # IC -> thalamic matrix
        netParams.connParams['IC->ThalEMatrix'] = { 
            'preConds': {'pop': 'IC'}, 
            'postConds': {'pop': ['TCM']},
            'sec': 'soma', 
            'loc': 0.5,
            'synMech': ESynMech,
            'probability': cfg.ICThalInput['probEMatrix'],
            'weight': cfg.ICThalInput['weightEMatrix'],
            'synMechWeightFactor': cfg.synWeightFractionEE,
            'delay': cfg.delayBkg}        
        netParams.connParams['IC->ThalIMatrix'] = { 
            'preConds': {'pop': 'IC'}, 
            'postConds': {'pop': ['IREM', 'TIM']},
            'sec': 'soma', 
            'loc': 0.5,
            'synMech': 'GABAA',
            'probability': cfg.ICThalInput['probIMatrix'],
            'weight': cfg.ICThalInput['weightIMatrix'],
            'delay': cfg.delayBkg}  
        
def prob2conv (prob, npre):
  # probability to convergence; prob is connection probability, npre is number of presynaptic neurons
  return int(0.5 + prob * npre)

# cochlea -> thal
def connectCochleaToThal ():
  print('connecting cochlea to thal')
  # these next two parameters are derived, so should be set here in case used by batch/optimization; because cfg.py
  # gets copied as a .json file without re-interpreting the other variables
  cfg.cochlearThalInput['weightEMatrix'] = cfg.cochlearThalInput['weightECore'] * cfg.cochlearThalInput['MatrixCoreFactor']
  cfg.cochlearThalInput['weightIMatrix'] = cfg.cochlearThalInput['weightICore'] * cfg.cochlearThalInput['MatrixCoreFactor']
  # cochlea to thalamic core uses topographic wiring, cochlea to matrix uses random wiring
  for ct in ['TC', 'HTC']:  # cochlea -> Thal Core E neurons
    prob = '%f * exp(-dist_x/%f)' % (cfg.cochlearThalInput['probECore'], dconf['net']['ThalamicCoreLambda'])
    # prob = cfg.cochlearThalInput['probECore'] # debug
    print('coch to thal prob:', prob)
    netParams.connParams['cochlea->ThalECore'+ct] = { 
        'preConds': {'pop': 'cochlea'}, 
        'postConds': {'cellType': [ct]},
        'sec': 'soma', 
        'loc': 0.5,
        'synMech': ESynMech,
        'probability': prob,
        'weight': cfg.cochlearThalInput['weightECore'],
        'synMechWeightFactor': cfg.synWeightFractionEE,
        'delay': cfg.delayBkg}
    print('np:',netParams.connParams['cochlea->ThalECore'+ct])
  for ct in ['IRE', 'TI']:
    prob = '%f * exp(-dist_x/%f)' % (cfg.cochlearThalInput['probICore'],dconf['net']['ThalamicCoreLambda'])
    # prob = cfg.cochlearThalInput['probICore'] # debug
    print('coch to thal prob:', prob)    
    netParams.connParams['cochlea->ThalICore'+ct] = { 
        'preConds': {'pop': 'cochlea'}, 
        'postConds': {'cellType': [ct]},
        'sec': 'soma', 
        'loc': 0.5,
        'synMech': ESynMech,
        'probability': prob,
        'weight': cfg.cochlearThalInput['weightICore'],
        'synMechWeightFactor': cfg.synWeightFractionEI,
        'delay': cfg.delayBkg}  
  # cochlea -> Thal Matrix
  netParams.connParams['cochlea->ThalEMatrix'] = { 
      'preConds': {'pop': 'cochlea'}, 
      'postConds': {'cellType': ['TCM']},
      'sec': 'soma', 
      'loc': 0.5,
      'synMech': ESynMech,
      'convergence': prob2conv(cfg.cochlearThalInput['probEMatrix'], numCochlearCells), 
      'weight': cfg.cochlearThalInput['weightEMatrix'],
      'synMechWeightFactor': cfg.synWeightFractionEE,
      'delay': cfg.delayBkg}
  netParams.connParams['cochlea->ThalIMatrix'] = { 
      'preConds': {'pop': 'cochlea'}, 
      'postConds': {'cellType': ['IREM','TIM']},
      'sec': 'soma', 
      'loc': 0.5,
      'synMech': ESynMech,
      'convergence': prob2conv(cfg.cochlearThalInput['probIMatrix'], numCochlearCells),
      'weight': cfg.cochlearThalInput['weightIMatrix'],
      'synMechWeightFactor': cfg.synWeightFractionEI,
      'delay': cfg.delayBkg}

if cfg.cochlearThalInput: connectCochleaToThal()        

#------------------------------------------------------------------------------
# Current inputs (IClamp)
#------------------------------------------------------------------------------
# if cfg.addIClamp:
#  	for key in [k for k in dir(cfg) if k.startswith('IClamp')]:
# 		params = getattr(cfg, key, None)
# 		[pop,sec,loc,start,dur,amp] = [params[s] for s in ['pop','sec','loc','start','dur','amp']]
        
#         		# add stim source
# 		netParams.stimSourceParams[key] = {'type': 'IClamp', 'delay': start, 'dur': dur, 'amp': amp}
        
# 		# connect stim source to target
# 		netParams.stimTargetParams[key+'_'+pop] =  {
# 			'source': key, 
# 			'conds': {'pop': pop},
# 			'sec': sec, 
# 			'loc': loc}

#------------------------------------------------------------------------------
# NetStim inputs (to simulate short external stimuli; not bkg)
#------------------------------------------------------------------------------
if cfg.addNetStim:
    for key in [k for k in dir(cfg) if k.startswith('NetStim')]:
        params = getattr(cfg, key, None)
        [pop, ynorm, sec, loc, synMech, synMechWeightFactor, start, interval, noise, number, weight, delay] = \
        [params[s] for s in ['pop', 'ynorm', 'sec', 'loc', 'synMech', 'synMechWeightFactor', 'start', 'interval', 'noise', 'number', 'weight', 'delay']] 

        # add stim source
        netParams.stimSourceParams[key] = {'type': 'NetStim', 'start': start, 'interval': interval, 'noise': noise, 'number': number}
        
        if not isinstance(pop, list):
            pop = [pop]

        for eachPop in pop:
            # connect stim source to target 
            print(key, eachPop)
            netParams.stimTargetParams[key+'_'+eachPop] =  {
                'source': key, 
                'conds': {'pop': eachPop, 'ynorm': ynorm},
                'sec': sec, 
                'loc': loc,
                'synMech': synMech,
                'weight': weight,
                'synMechWeightFactor': synMechWeightFactor,
                'delay': delay}

#------------------------------------------------------------------------------
# Description
#------------------------------------------------------------------------------

netParams.description = """
v7 - Added template for connectivity
v8 - Added cell types
v9 - Added local connectivity
v10 - Added thalamic populations from prev model
v11 - Added thalamic conn from prev model
v12 - Added CT cells to L5B
v13 - Added CT cells to L5A
v14 - Fixed L5A & L5B E cell densities + added CT5A & CT5B to 'Epops'
v15 - Added cortical and thalamic conn to CT5A and CT5B 
v16 - Updated multiple cell types
v17 - Changed NGF -> I prob from strong (1.0) to weak (0.35)
v18 - Fixed bug in VIP cell morphology
v19 - Added in 2-compartment thalamic interneuron model 
v20 - Added TI conn and updated thal pop
v21 - Added exc+inh bkg inputs specific to each cell type
v22 - Made exc+inh bkg inputs specific to each pop; automated calculation
v23 - IE/II specific layer gains and simplified code (assume 'Allen_custom')
v24 - Fixed bug in IE/II specific layer gains
v25 - Fixed subconnparams TC->E and NGF1->E; made IC input deterministic
v26 - Changed NGF AMPA:NMDA ratio 
v27 - Split thalamic interneurons into core and matrix (TI and TIM)
v28 - Set recurrent TC->TC conn to 0
v29 - Added EI specific layer gains
v30 - Added EE specific layer gains; and split combined L1-3 gains into L1,L2,L3
v31 - Added EI postsyn-cell-type specific gains; update ITS4 and NGF
v32 - Added IE presyn-cell-type specific gains
v33 - Fixed bug in matrix thalamocortical conn (were very low)
v34 - Added missing conn from cortex to matrix thalamus IREM and TIM
v35 - Parametrize L5B PT Ih and exc cell K+ conductance (to simulate NA/ACh modulation) 
v36 - Looped speech stimulus capability added for cfg.ICThalInput
v37 - Adding in code to modulate t-type calcium conductances in thalamic and cortical cells
v38 - Adding in code to modulate NMDA synaptic weight from E --> I populations 
v39 - Changed E --> I cfg.NMDARfactor such that weight is not a list, but instead a single value 
v40 - added parameterizations from Christoph Metzner for localizing the large L1 sink
v41 - modifying cochlea to Thal -> A1 for tonotopic gradient, adding functions
"""

"""
init.py

Starting script to run NetPyNE-based A1 model.


Usage:
    python init.py # Run simulation, optionally plot a raster


MPI usage:
    mpiexec -n 4 nrniv -python -mpi init.py


Contributors: ericaygriffith@gmail.com, salvadordura@gmail.com
"""

onServer = False

import matplotlib
if onServer: matplotlib.use('Agg')  # to avoid graphics error in servers

# sim.createSimulateAnalyze(netParams, cfg)
####### UNCOMMENTED THE ABOVE FOR TEST TRIAL RUN ON NEUROSIM 

sim.initialize(simConfig=cfg, netParams=netParams)	# create network object and set cfg and net params

from utils import backupcfg, safemkdir
backupcfg(dconf['sim']['name'])
safemkdir('data') # make sure data (output) directory exists

sim.net.createPops()               			# instantiate network populations
sim.net.createCells()              			# instantiate network cells based on defined populations

def setdminID (sim, lpop):
  # setup min,max ID and dnumc for each population in lpop
  alltags = sim._gatherAllCellTags() #gather cell tags; see https://github.com/Neurosim-lab/netpyne/blob/development/netpyne/sim/gather.py
  dGIDs = {pop:[] for pop in lpop}
  for tinds in range(len(alltags)):
    if alltags[tinds]['pop'] in lpop:
      dGIDs[alltags[tinds]['pop']].append(tinds)
  sim.simData['dminID'] = {pop:np.amin(dGIDs[pop]) for pop in lpop if len(dGIDs[pop])>0}
  sim.simData['dmaxID'] = {pop:np.amax(dGIDs[pop]) for pop in lpop if len(dGIDs[pop])>0}
  sim.simData['dnumc'] = {pop:np.amax(dGIDs[pop])-np.amin(dGIDs[pop]) for pop in lpop if len(dGIDs[pop])>0}

setdminID(sim, cfg.allpops)

# print(sim.simData['dminID'])

def setCochCellLocationsX (pop, sz, scale):
  # set the cell positions on a line
  if pop not in sim.net.pops: return
  offset = sim.simData['dminID'][pop]
  ncellinrange = 0 # number of cochlear cells with center frequency in frequency range represented by this model
  sidx = -1
  for idx,cf in enumerate(cochlearCenterFreqs):
    if cf >= cfg.cochThalFreqRange[0] and cf <= cfg.cochThalFreqRange[1]:
      if sidx == -1: sidx = idx # start index
      ncellinrange += 1
  if sidx > -1: offset += sidx
  print('setCochCellLocations: sidx, offset, ncellinrange = ', sidx, offset, ncellinrange)
  for c in sim.net.cells:
    if c.gid in sim.net.pops[pop].cellGids:
      cf = cochlearCenterFreqs[c.gid-sim.simData['dminID'][pop]]
      if cf >= cfg.cochThalFreqRange[0] and cf <= cfg.cochThalFreqRange[1]:
        c.tags['x'] = cellx = ((c.gid-offset)/ncellinrange) * scale
        c.tags['xnorm'] = cellx / netParams.sizeX # make sure these values consistent
        print('gid,cellx,xnorm,cf=',c.gid,cellx,cellx/netParams.sizeX,cf)
      else:
        c.tags['x'] = cellx = 100000  # put it outside range for core
        c.tags['xnorm'] = cellx / netParams.sizeX # make sure these values consistent
      c.updateShape()

if dconf['sim']['useCochleaThal']: setCochCellLocationsX('cochlea', numCochlearCells, dconf['net']['sizeX'])

lsynweights = []

def recordWeightsPop (sim, prename, postname):
  print('recordWeightsPop',prename,postname)
  # record the weights for specified prename population to postname population
  lcell = [c for c in sim.net.cells if c.gid in sim.net.pops[postname].cellGids] # this is the set of MR cells
  print(postname,'len(lcell)=',len(lcell))
  for cell in lcell:
    print('len(cell.conns)=',len(cell.conns))
    for conn in cell.conns:
      if type(conn.preGid)!=int and type(conn.preGid)!=float:
        # print('conn.preGid not int or float',conn.preGid)
        continue
      if conn.preGid >= sim.simData['dminID'][prename] and conn.preGid <= sim.simData['dmaxID'][prename]:
        # print('found conn from', prename, 'to', postname)
        lsynweights.append([conn.preGid,cell.gid,float(conn['hObj'].weight[0])])
  return len(lcell)

def checkCochPreGids ():
  print('checkCochPreGids()')
  cochGids = []
  cochConns = []
  for cell in sim.net.cells:
    if cell.tags['pop'] == 'cochlea':
      cochGids.append(cell.gid)
  for cell in sim.net.cells:
    for conn in cell.conns:
      if conn['preGid'] in cochGids:
        cochConns.append(conn)
  print('found ', len(cochConns), ' source cochlear synaptic connections')  

def LSynWeightToD (L):
  # convert list of synaptic weights to dictionary to save disk space
  print('converting synaptic weight list to dictionary...')
  dout = {}; 
  for row in L:
    #t,preID,poID,w,cumreward = row
    preID,poID,w = row
    if preID not in dout:
      dout[preID] = {}
    if poID not in dout[preID]:
      dout[preID][poID] = []
    dout[preID][poID].append([w])
  return dout

def saveSynWeights ():
  # save synaptic weights 
  fn = 'data/'+dconf['sim']['name']+'synWeights_'+str(sim.rank)+'.pkl'
  pickle.dump(lsynweights, open(fn, 'wb')) # save synaptic weights to disk for this node
  sim.pc.barrier() # wait for other nodes
  time.sleep(3)    
  if sim.rank == 0: # rank 0 reads and assembles the synaptic weights into a single output file
    L = []
    for i in range(sim.nhosts):
      fn = 'data/'+dconf['sim']['name']+'synWeights_'+str(i)+'.pkl'
      while not os.path.isfile(fn): # wait until the file is written/available
        print('saveSynWeights: waiting for finish write of', fn)
        time.sleep(3)
      lw = pickle.load(open(fn,'rb'))
      print(fn,'len(lw)=',len(lw),type(lw))
      os.unlink(fn) # remove the temporary file
      L = L + lw # concatenate to the list L
    #pickle.dump(L,open('data/'+dconf['sim']['name']+'synWeights.pkl', 'wb')) # this would save as a List
    # now convert the list to a dictionary to save space, and save it to disk
    dout = LSynWeightToD(L)
    pickle.dump(dout,open('data/'+dconf['sim']['name']+'synWeights.pkl', 'wb'))


sim.net.connectCells()            			# create connections between cells based on params
sim.net.addStims() 							# add network stimulation
sim.setupRecording()              			# setup variables to record for each cell (spikes, V traces, etc)

if dconf['sim']['useCochleaThal']:
  print('useCochleaThal')
  recordWeightsPop(sim,'cochlea','TC')
  checkCochPreGids()
saveSynWeights()

sim.runSim()                      			# run parallel Neuron simulation  

# sim.gatherData()                  			# gather spiking data and cell info from each node
# sim.saveData()                    			# save params, cell info and sim output to file (pickle,mat,txt,etc)#

# if sim.rank==0: sim.analysis.plotData()         	# plot spike raster etc

# distributed saving (to avoid errors with large output data)
if dconf['sim']['dosave']:
  sim.saveDataInNodes()
  sim.gatherDataFromFiles(saveMerged=True)

'''
if sim.rank == 0:
    layer_bounds= {'L1': 100, 'L2': 160, 'L3': 950, 'L4': 1250, 'L5A': 1334, 'L5B': 1550, 'L6': 2000}
    filename = sim.cfg.saveFolder+'/'+sim.cfg.simLabel
    sim.analysis.plotRaster(**{'include': sim.cfg.allpops, 'saveFig': filename+'_1sec', 'showFig': False, 'popRates': True, 'orderInverse': True, 'timeRange': [1500,2500], 'figSize': (14,12), 'lw': 0.3, 'markerSize': 3, 'marker': '.', 'dpi': 300})
    sim.analysis.plotRaster(**{'include': sim.cfg.allpops, 'saveFig': filename+'_5sec', 'showFig': False, 'popRates': True, 'orderInverse': True, 'timeRange': [1500,6500], 'figSize': (14,12), 'lw': 0.3, 'markerSize': 3, 'marker': '.', 'dpi': 300})
    sim.analysis.plotRaster(**{'include': sim.cfg.allpops, 'saveFig': filename+'_10sec', 'showFig': False, 'popRates': True, 'orderInverse': True, 'timeRange': [1500,11500], 'figSize': (14,12), 'lw': 0.3, 'markerSize': 3, 'marker': '.', 'dpi': 300})
    #sim.analysis.plotSpikeStats(stats=['rate'],figSize = (6,12), timeRange=[1500, 11500], dpi=300, showFig=0, saveFig=1)        
    for elec in [1, 6, 11, 16]:
        sim.analysis.plotLFP(**{'plots': ['timeSeries'], 'electrodes': [elec], 'timeRange': [1500, 6500], 'maxFreq':80, 'figSize': (8,4), 'saveData': False, 'saveFig': filename+'_LFP_signal_5s_elec_'+str(elec), 'showFig': False})
        sim.analysis.plotLFP(**{'plots': ['PSD'], 'electrodes': [elec], 'timeRange': [1500, 2500], 'maxFreq':80, 'figSize': (8,4), 'saveData': False, 'saveFig': filename+'_LFP_PSD_5s_elec_'+str(elec), 'showFig': False})
        sim.analysis.plotLFP(**{'plots': ['spectrogram'], 'electrodes': [elec], 'timeRange': [1500, 6500], 'maxFreq':80, 'figSize': (8,4), 'saveData': False, 'saveFig': filename+'_LFP_spec_5s_elec_'+str(elec), 'showFig': False})
        sim.analysis.plotLFP(**{'plots': ['timeSeries'], 'electrodes': [elec], 'timeRange': [1500, 11500], 'maxFreq':80, 'figSize': (16,4), 'saveData': False, 'saveFig': filename+'_LFP_signal_10s_elec_'+str(elec), 'showFig': False})
        sim.analysis.plotLFP(**{'plots': ['PSD'], 'electrodes': [elec], 'timeRange': [1500, 11500], 'maxFreq':80, 'figSize': (16,4), 'saveData': False, 'saveFig': filename+'_LFP_PSD_10s_elec_'+str(elec), 'showFig': False})
        sim.analysis.plotLFP(**{'plots': ['spectrogram'], 'electrodes': [elec], 'timeRange': [1500, 11500], 'maxFreq':80, 'figSize': (16,4), 'saveData': False, 'saveFig': filename+'_LFP_spec_10s_elec_'+str(elec), 'showFig': False})
    # tranges = [[2000, 2200],
    #             [2000, 2100]]
    #     #         [1980, 2100],
    #     #         [2080, 2200],
    #     #         [2030, 2150],
    #     #         [2100, 2200]] 
    # for t in tranges:
    #     sim.analysis.plotCSD(**{'spacing_um': 100, 'overlay': 'LFP', 'layer_lines': 1, 'layer_bounds': layer_bounds, 'timeRange': [t[0], t[1]], 'saveFig': filename[:-4]+'_CSD_%d-%d' % (t[0], t[1]), 'figSize': (6,9), 'dpi': 300, 'showFig': 0})
'''

if (sim.rank == 0 and sim.pc.nhost()>1) or dconf['sim']['doquit']: quit()
