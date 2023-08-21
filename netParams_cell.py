"""
netParams.py 

High-level specifications for A1 network model using NetPyNE

Contributors: ericaygriffith@gmail.com, salvadordura@gmail.com
"""

from netpyne import specs
import pickle, json

netParams = specs.NetParams()   # object of class NetParams to store the network parameters

try:
    from __main__ import cfg  # import SimConfig object with params from parent module
except:
    from cfg_cell import cfg


#------------------------------------------------------------------------------
# VERSION 
#------------------------------------------------------------------------------
netParams.version = 29

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
netParams.scaleConnWeightModels = {'HH_reduced': 1.0, 'HH_reduced': 1.0, 'HH_full': 1.0} #scale conn weight factor for each cell model
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
layer = {'1': [0.00, 0.05], '2': [0.05, 0.08], '3': [0.08, 0.475], '4': [0.475, 0.625], '5A': [0.625, 0.667], '5B': [0.667, 0.775], '6': [0.775, 1], 'thal': [1.2, 1.4]}  # normalized layer boundaries  

# add layer border correction ??
#netParams.correctBorder = {'threshold': [cfg.correctBorderThreshold, cfg.correctBorderThreshold, cfg.correctBorderThreshold], 
#                        'yborders': [layer['2'][0], layer['5A'][0], layer['6'][0], layer['6'][1]]}  # correct conn border effect

#------------------------------------------------------------------------------
## Load cell rules previously saved using netpyne format (DOES NOT INCLUDE VIP, NGF and spiny stellate)
## include conditions ('conds') for each cellRule
cellParamLabels = { 'IT2_reduced':  {'cellModel': 'HH_reduced', 'cellType': 'IT', 'ynorm': layer['2']},
                    'IT3_reduced':  {'cellModel': 'HH_reduced', 'cellType': 'IT', 'ynorm': layer['3']},
                    'ITP4_reduced': {'cellModel': 'HH_reduced', 'cellType': 'IT', 'ynorm': layer['4']},
                    'IT5A_reduced': {'cellModel': 'HH_reduced', 'cellType': 'IT', 'ynorm': layer['5A']},
                    'CT5A_reduced': {'cellModel': 'HH_reduced', 'cellType': 'CT', 'ynorm': layer['5A']},
                    'IT5B_reduced': {'cellModel': 'HH_reduced', 'cellType': 'IT', 'ynorm': layer['5B']},
                    'PT5B_reduced': {'cellModel': 'HH_reduced', 'cellType': 'PT', 'ynorm': layer['5B']},
                    'CT5B_reduced': {'cellModel': 'HH_reduced', 'cellType': 'CT', 'ynorm': layer['5B']},
                    'IT6_reduced':  {'cellModel': 'HH_reduced', 'cellType': 'IT', 'ynorm': layer['6']},
                    'CT6_reduced':  {'cellModel': 'HH_reduced', 'cellType': 'CT', 'ynorm': layer['6']},
                    'PV_reduced':  {'cellModel': 'HH_reduced', 'cellType': 'PV', 'ynorm': [layer['2'][0],layer['6'][1]]},
                    'SOM_reduced': {'cellModel': 'HH_reduced', 'cellType': 'SOM', 'ynorm': [layer['2'][0], layer['6'][1]]}}

## Import VIP cell rule from hoc file 
netParams.importCellParams(label='VIP_reduced', conds={'cellType': 'VIP', 'cellModel': 'HH_reduced'}, fileName='cells/vipcr_cell.hoc', cellName='VIPCRCell_EDITED', importSynMechs=True)
netParams.cellParams['VIP_reduced']['conds'] = {'cellModel': 'HH_reduced', 'cellType': 'VIP', 'ynorm': [layer['2'][0], layer['6'][1]]}

## Import NGF cell rule from hoc file
netParams.importCellParams(label='NGF_reduced', conds={'cellType': 'NGF', 'cellModel': 'HH_reduced'}, fileName='cells/ngf_cell.hoc', cellName='ngfcell', importSynMechs=True)
netParams.cellParams['NGF_reduced']['conds'] = {'cellModel': 'HH_reduced', 'cellType': 'NGF', 'ynorm': [layer['1'][0], layer['6'][1]]}

## Import L4 Spiny Stellate cell rule from .py file
netParams.importCellParams(label='ITS4_reduced', conds={'cellType': 'ITS4', 'cellModel': 'HH_reduced'}, fileName='cells/ITS4.py', cellName='ITS4_cell')
netParams.cellParams['ITS4_reduced']['conds'] = {'cellModel': 'HH_reduced', 'cellType': 'ITS4', 'ynorm': layer['4']}

## THALAMIC CELL MODELS

# Import RE (reticular) cell rule from .py file 
netParams.importCellParams(label='RE_reduced', conds={'cellType': 'RE', 'cellModel': 'HH_reduced'}, fileName='cells/sRE.py', cellName='sRE', importSynMechs=True)
netParams.cellParams['RE_reduced']['conds'] = {'cellModel': 'HH_reduced', 'cellType': 'RE', 'ynorm': layer['thal']}

# Import TC cell rule from .py file 
netParams.importCellParams(label='TC_reduced', conds={'cellType': 'TC', 'cellModel': 'HH_reduced'}, fileName='cells/sTC.py', cellName='sTC', importSynMechs=True)
netParams.cellParams['TC_reduced']['conds'] = {'cellModel': 'HH_reduced', 'cellType': 'TC', 'ynorm': layer['thal']}

# Import HTC cell rule from .py file 
netParams.importCellParams(label='HTC_reduced', conds={'cellType': 'HTC', 'cellModel': 'HH_reduced'}, fileName='cells/sHTC.py', cellName='sHTC', importSynMechs=True)
netParams.cellParams['HTC_reduced']['conds'] = {'cellModel': 'HH_reduced', 'cellType': 'HTC', 'ynorm': layer['thal']}

# Import Thalamic Interneuron cell from .py file 
netParams.importCellParams(label='TI_reduced', conds={'cellType': 'TI', 'cellModel': 'HH_reduced'}, fileName='cells/sTI.py', cellName='sTI_cell', importSynMechs=True)
netParams.cellParams['TI_reduced']['conds'] = {'cellModel': 'HH_reduced', 'cellType': 'TI', 'ynorm': layer['thal']}

# Load cell rules from .pkl / .json file 
cellParamLabels = ['IT2_reduced', 'IT3_reduced', 'ITP4_reduced', 'ITS4_reduced',
                    'IT5A_reduced', 'CT5A_reduced', 'IT5B_reduced',
                    'PT5B_reduced', 'CT5B_reduced', 'IT6_reduced', 'CT6_reduced',
                    'PV_reduced', 'SOM_reduced', 'VIP_reduced', 'NGF_reduced',
                    'RE_reduced', 'TC_reduced', 'HTC_reduced', 'TI_reduced']

for ruleLabel in cellParamLabels:
    netParams.loadCellParamsRule(label=ruleLabel, fileName='cells/' + ruleLabel + '_cellParams.json')  # Load cellParams for each of the above cell subtype
    #netParams.cellParams[ruleLabel]['conds'] = cellParamLabels[ruleLabel]

## Options to add to cellParams
addSecLists = False
add3DGeom = False

## Set weightNorm for each cell type and add section lists (used in connectivity)
for ruleLabel in netParams.cellParams.keys():
    try:
        netParams.addCellParamsWeightNorm(ruleLabel, 'cells/' + ruleLabel + '_weightNorm.pkl', threshold=cfg.weightNormThreshold)  # add weightNorm
        print('   Loaded weightNorm pkl file for %s...' % (ruleLabel))
    except:
        print('   No weightNorm pkl file for %s...' % (ruleLabel))

    # remove
    if cfg.removeWeightNorm:
        for sec in netParams.cellParams[ruleLabel]['secs']:
            if 'weightNorm' in netParams.cellParams[ruleLabel]['secs'][sec]:    
                del netParams.cellParams[ruleLabel]['secs'][sec]['weightNorm']

    if addSecLists:
        secLists = {}
        if ruleLabel in ['IT2_reduced', 'IT3_reduced', 'ITP4_reduced', 'IT5A_reduced', 'CT5A_reduced', 'IT5B_reduced', 'PT5B_reduced', 'CT5B_reduced', 'IT6_reduced', 'CT6_reduced']:
            secLists['all'] = ['soma', 'Adend1', 'Adend2', 'Adend3', 'Bdend']
            secLists['proximal'] = ['soma', 'Bdend', 'Adend1']
            secLists['dend_all'] = ['Adend1', 'Adend2', 'Adend3', 'Bdend']
            secLists['apic'] = ['Adend1', 'Adend2', 'Adend3']
            secLists['apic_trunk'] = ['Adend1', 'Adend2']
            secLists['apic_lowertrunk'] = ['Adend1']
            secLists['apic_uppertrunk'] = ['Adend2']
            secLists['apic_tuft'] = ['Adend3']

        elif ruleLabel in ['ITS4_reduced']:
            secLists['all'] = secLists['proximal'] = ['soma', 'dend', 'dend1']
            secLists['dend_all'] = secLists['apic'] = secLists['apic_trunk'] = secLists['apic_lowertrunk'] = \
                secLists['apic_uppertrunk'] = secLists['apic_tuft'] = ['dend', 'dend1']

        elif ruleLabel in ['PV_reduced', 'SOM_reduced', 'NGF_reduced', 'TI_reduced']:
            secLists['all'] = secLists['proximal'] = ['soma', 'dend']
            secLists['dend_all'] = ['dend']

        elif ruleLabel in ['VIP_reduced']:
            secLists['all'] = ['soma', 'rad1', 'rad2', 'ori1', 'ori2']
            secLists['proximal'] = ['soma', 'rad1', 'ori1']
            secLists['dend_all'] = ['rad1', 'rad2', 'ori1', 'ori2']

        # store secLists in netParams
        netParams.cellParams[ruleLabel]['secLists'] = dict(secLists)


if add3DGeom:
    ## Set 3D geometry for each cell type
    for label in netParams.cellParams:
        if label in ['PV_reduced', 'SOM_reduced']: 
            offset, prevL = 0, 0
            somaL = netParams.cellParams[label]['secs']['soma']['geom']['L']
            for secName in ['soma', 'dend', 'axon']: 
                sec = netParams.cellParams[label]['secs'][secName]
                sec['geom']['pt3d'] = []
                if secName in ['soma', 'dend']:  # set 3d geom of soma and Adends                    
                    sec['geom']['pt3d'].append([offset+0, prevL, 0, sec['geom']['diam']])
                    prevL = float(prevL + sec['geom']['L'])
                    sec['geom']['pt3d'].append([offset + 0, prevL, 0, sec['geom']['diam']])
                    print(label, secName, sec['geom']['pt3d'])
                if secName in ['axon']:  # set 3d geom of axon
                    sec['geom']['pt3d'].append([offset+0, 0, 0, sec['geom']['diam']])
                    sec['geom']['pt3d'].append([offset + 0, -sec['geom']['L'], 0, sec['geom']['diam']])

        elif label in ['NGF_reduced', 'TI_reduced']:
            offset, prevL = 0, 0
            somaL = netParams.cellParams[label]['secs']['soma']['geom']['L']
            for secName in ['soma', 'dend']:
                sec = netParams.cellParams[label]['secs'][secName]
                sec['geom']['pt3d'] = []
                if secName in ['soma', 'dend']:  # set 3d geom of soma and Adends
                    sec['geom']['pt3d'].append([offset+0, prevL, 0, sec['geom']['diam']])
                    prevL = float(prevL + sec['geom']['L'])
                    sec['geom']['pt3d'].append([offset + 0, prevL, 0, sec['geom']['diam']])

        elif label in ['VIP_reduced']:
            offset, prevL = 0, 0
            somaL = netParams.cellParams[label]['secs']['soma']['geom']['L']
            for secName in ['soma', 'rad1', 'rad2', 'ori1', 'ori2']:
                sec = netParams.cellParams[label]['secs'][secName]
                sec['geom']['pt3d'] = []
                if secName in ['soma']:  # set 3d geom of soma 
                    sec['geom']['pt3d'].append([offset+0, prevL, 0, sec['geom']['diam']])
                    prevL = float(prevL + sec['geom']['L'])
                    sec['geom']['pt3d'].append([offset + 0, prevL, 0, sec['geom']['diam']])
                if secName in ['rad1']:  # set 3d geom of rad1 (radiatum)
                    sec['geom']['pt3d'].append([offset+0, somaL, 0, sec['geom']['diam']])
                    sec['geom']['pt3d'].append([offset+0.5*sec['geom']['L'], +(somaL+0.866*sec['geom']['L']), 0, sec['geom']['diam']])   
                if secName in ['rad2']:  # set 3d geom of rad2 (radiatum)
                    sec['geom']['pt3d'].append([offset+0, somaL, 0, sec['geom']['diam']])
                    sec['geom']['pt3d'].append([offset-0.5*sec['geom']['L'], +(somaL+0.866*sec['geom']['L']), 0, sec['geom']['diam']])   
                if secName in ['ori1']:  # set 3d geom of ori1 (oriens)
                    sec['geom']['pt3d'].append([offset+0, somaL, 0, sec['geom']['diam']])
                    sec['geom']['pt3d'].append([offset+0.707*sec['geom']['L'], -(somaL+0.707*sec['geom']['L']), 0, sec['geom']['diam']])   
                if secName in ['ori2']:  # set 3d geom of ori2 (oriens)
                    sec['geom']['pt3d'].append([offset+0, somaL, 0, sec['geom']['diam']])
                    sec['geom']['pt3d'].append([offset-0.707*sec['geom']['L'], -(somaL+0.707*sec['geom']['L']), 0, sec['geom']['diam']])   


        elif label in ['ITS4_reduced']:
            offset, prevL = 0, 0
            somaL = netParams.cellParams[label]['secs']['soma']['geom']['L']
            for secName in ['soma', 'dend', 'dend1']:
                sec = netParams.cellParams[label]['secs'][secName]
                sec['geom']['pt3d'] = []
                if secName in ['soma']:  # set 3d geom of soma 
                    sec['geom']['pt3d'].append([offset+0, prevL, 0, sec['geom']['diam']])
                    prevL = float(prevL + sec['geom']['L'])
                    sec['geom']['pt3d'].append([offset + 0, prevL, 0, sec['geom']['diam']])
                if secName in ['dend']:  # set 3d geom of apic dendds
                    sec['geom']['pt3d'].append([offset+0, prevL, 0, sec['geom']['diam']])
                    prevL = float(prevL + sec['geom']['L'])
                    sec['geom']['pt3d'].append([offset + 0, prevL, 0, sec['geom']['diam']])
                if secName in ['dend1']:  # set 3d geom of basal dend
                    sec['geom']['pt3d'].append([offset+0, somaL, 0, sec['geom']['diam']])
                    sec['geom']['pt3d'].append([offset+0.707*sec['geom']['L'], -(somaL+0.707*sec['geom']['L']), 0, sec['geom']['diam']])   
        
        elif label in ['RE_reduced', 'TC_reduced', 'HTC_reduced', 'TI_reduced']:
            sec = netParams.cellParams[label]['secs']['soma']
            sec['geom']['pt3d'] = []
            sec['geom']['pt3d'].append([offset+0, 0, 0, sec['geom']['diam']])
            sec['geom']['pt3d'].append([offset+0, sec['geom']['L'], 0, sec['geom']['diam']])

        else: # E cells
            # set 3D pt geom
            offset, prevL = 0, 0
            somaL = netParams.cellParams[label]['secs']['soma']['geom']['L']
            for secName in ['soma', 'Adend1', 'Adend2', 'Adend3', 'Bdend', 'axon']:
                sec = netParams.cellParams[label]['secs'][secName]
                sec['geom']['pt3d'] = []
                if secName in ['soma', 'Adend1', 'Adend2', 'Adend3']:  # set 3d geom of soma and Adends
                    sec['geom']['pt3d'].append([offset+0, prevL, 0, sec['geom']['diam']])
                    prevL = float(prevL + sec['geom']['L'])
                    sec['geom']['pt3d'].append([offset+0, prevL, 0, sec['geom']['diam']])
                if secName in ['Bdend']:  # set 3d geom of Bdend
                    sec['geom']['pt3d'].append([offset+0, somaL, 0, sec['geom']['diam']])
                    sec['geom']['pt3d'].append([offset+0.707*sec['geom']['L'], -(somaL+0.707*sec['geom']['L']), 0, sec['geom']['diam']])        
                if secName in ['axon']:  # set 3d geom of axon
                    sec['geom']['pt3d'].append([offset+0, 0, 0, sec['geom']['diam']])
                    sec['geom']['pt3d'].append([offset + 0, -sec['geom']['L'], 0, sec['geom']['diam']])
                    

# save cellParams rules to .pkl file
saveCellParams = True
if saveCellParams:
    for ruleLabel in netParams.cellParams.keys():
        netParams.saveCellParamsRule(label=ruleLabel, fileName='cells/' + ruleLabel + '_cellParams.json')


#------------------------------------------------------------------------------
# Population parameters
#------------------------------------------------------------------------------

## load densities
with open('cells/cellDensity.pkl', 'rb') as fileObj: density = pickle.load(fileObj)['density']
density = {k: [x * cfg.scaleDensity for x in v] for k,v in density.items()} # Scale densities 

### LAYER 1:
netParams.popParams['NGF1'] = {'cellType': 'NGF', 'cellModel': 'HH_reduced','ynormRange': layer['1'],   'density': density[('A1','nonVIP')][0]}

##LAYER 2:
netParams.popParams['IT2'] =    {'cellType': 'IT',   'cellModel':  'HH_reduced',  'ynormRange': layer['2'],   'density': density[('A1', 'E')][1]}
netParams.popParams['SOM2'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',   'ynormRange': layer['2'],   'density': density[('A1','SOM')][1]}   
netParams.popParams['PV2'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',   'ynormRange': layer['2'],   'density': density[('A1','PV')][1]}    
netParams.popParams['VIP2'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',   'ynormRange': layer['2'],   'density': density[('A1','VIP')][1]}
netParams.popParams['NGF2'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',   'ynormRange': layer['2'],   'density': density[('A1','nonVIP')][1]}

##LAYER 3:
netParams.popParams['IT3'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': layer['3'],   'density': density[('A1','E')][1]} 
netParams.popParams['SOM3'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',   'ynormRange': layer['3'],   'density': density[('A1','SOM')][1]} 
netParams.popParams['PV3'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',   'ynormRange': layer['3'],   'density': density[('A1','PV')][1]} 
netParams.popParams['VIP3'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',   'ynormRange': layer['3'],   'density': density[('A1','VIP')][1]} 
netParams.popParams['NGF3'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',   'ynormRange': layer['3'],   'density': density[('A1','nonVIP')][1]}


## LAYER 4: 
netParams.popParams['ITP4'] =	 {'cellType': 'IT', 'cellModel': 'HH_reduced',  'ynormRange': layer['4'],   'density': 0.5*density[('A1','E')][2]}      
netParams.popParams['ITS4'] =	 {'cellType': 'ITS4', 'cellModel': 'HH_reduced', 'ynormRange': layer['4'],  'density': 0.5*density[('A1','E')][2]}       
netParams.popParams['SOM4'] = 	 {'cellType': 'SOM', 'cellModel': 'HH_reduced',   'ynormRange': layer['4'],  'density': density[('A1','SOM')][2]}
netParams.popParams['PV4'] = 	 {'cellType': 'PV', 'cellModel': 'HH_reduced',   'ynormRange': layer['4'],   'density': density[('A1','PV')][2]}
netParams.popParams['VIP4'] =	 {'cellType': 'VIP', 'cellModel': 'HH_reduced',   'ynormRange': layer['4'],  'density': density[('A1','VIP')][2]}
netParams.popParams['NGF4'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',   'ynormRange': layer['4'],  'density': density[('A1','nonVIP')][2]}

## LAYER 5A: 
netParams.popParams['IT5A'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5A'], 	'density': 0.5*density[('A1','E')][3]}      
netParams.popParams['CT5A'] =     {'cellType': 'CT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5A'],   'density': 0.5*density[('A1','E')][3]}  # density is [5] because we are using same numbers for L5A and L6 for CT cells? 
netParams.popParams['SOM5A'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',    'ynormRange': layer['5A'],	'density': density[('A1','SOM')][3]}          
netParams.popParams['PV5A'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',    'ynormRange': layer['5A'],	'density': density[('A1','PV')][3]}         
netParams.popParams['VIP5A'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',    'ynormRange': layer['5A'],   'density': density[('A1','VIP')][3]}
netParams.popParams['NGF5A'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',    'ynormRange': layer['5A'],   'density': density[('A1','nonVIP')][3]}

## LAYER 5B: 
netParams.popParams['IT5B'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5B'], 	'density': (1/3)*density[('A1','E')][4]}  
netParams.popParams['CT5B'] =     {'cellType': 'CT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5B'],   'density': (1/3)*density[('A1','E')][4]}  
netParams.popParams['PT5B'] =     {'cellType': 'PT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5B'], 	'density': (1/3)*density[('A1','E')][4]}  
netParams.popParams['SOM5B'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',    'ynormRange': layer['5B'],   'density': density[('A1', 'SOM')][4]}
netParams.popParams['PV5B'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',    'ynormRange': layer['5B'],	'density': density[('A1','PV')][4]}     
netParams.popParams['VIP5B'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',    'ynormRange': layer['5B'],   'density': density[('A1','VIP')][4]}
netParams.popParams['NGF5B'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',    'ynormRange': layer['5B'],   'density': density[('A1','nonVIP')][4]}

## LAYER 6:
netParams.popParams['IT6'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': layer['6'],   'density': 0.5*density[('A1','E')][5]}  
netParams.popParams['CT6'] =     {'cellType': 'CT',  'cellModel': 'HH_reduced',  'ynormRange': layer['6'],   'density': 0.5*density[('A1','E')][5]} 
netParams.popParams['SOM6'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',   'ynormRange': layer['6'],   'density': density[('A1','SOM')][5]}   
netParams.popParams['PV6'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',   'ynormRange': layer['6'],   'density': density[('A1','PV')][5]}     
netParams.popParams['VIP6'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',   'ynormRange': layer['6'],   'density': density[('A1','VIP')][5]}
netParams.popParams['NGF6'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',   'ynormRange': layer['6'],   'density': density[('A1','nonVIP')][5]}


## THALAMIC POPULATIONS (from prev model)
thalDensity = density[('A1','PV')][2] * 1.25  # temporary estimate (from prev model)

netParams.popParams['TC'] =     {'cellType': 'TC',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': 0.75*thalDensity}  
netParams.popParams['TCM'] =    {'cellType': 'TC',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': thalDensity} 
netParams.popParams['HTC'] =    {'cellType': 'HTC', 'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': 0.25*thalDensity}   
netParams.popParams['IRE'] =    {'cellType': 'RE',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': thalDensity}     
netParams.popParams['IREM'] =   {'cellType': 'RE', 'cellModel': 'HH_reduced',   'ynormRange': layer['thal'],   'density': thalDensity}
netParams.popParams['TI'] =     {'cellType': 'TI',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': 2*0.33 * thalDensity} ## Winer & Larue 1996; Huang et al 1999 


if cfg.singleCellPops:
    for popName,pop in netParams.popParams.items():
        if cfg.singlePop:
            if cfg.singlePop == popName:
                pop['numCells'] = 1
            else:
                pop['numCells'] = 0
        else:
            pop['numCells'] = 1

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
netParams.synMechParams['GABAB'] = {'mod':'MyExp2SynBB', 'tau1': 3.5, 'tau2': 260.9, 'e': -93} 
netParams.synMechParams['GABAA'] = {'mod':'MyExp2SynBB', 'tau1': 0.07, 'tau2': 18.2, 'e': -80}
netParams.synMechParams['GABAA_VIP'] = {'mod':'MyExp2SynBB', 'tau1': 0.3, 'tau2': 6.4, 'e': -80}  # Pi et al 2013
netParams.synMechParams['GABAASlow'] = {'mod': 'MyExp2SynBB','tau1': 2, 'tau2': 100, 'e': -80}
netParams.synMechParams['GABAASlowSlow'] = {'mod': 'MyExp2SynBB', 'tau1': 200, 'tau2': 400, 'e': -80}

ESynMech = ['AMPA', 'NMDA']
SOMESynMech = ['GABAASlow','GABAB']
SOMISynMech = ['GABAASlow']
PVSynMech = ['GABAA']
VIPSynMech = ['GABAA_VIP']
NGFSynMech = ['GABAA', 'GABAB']

'''
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

#------------------------------------------------------------------------------
## E -> E
if cfg.addConn:
    for pre in Epops:
        for post in Epops:
            if connDataSource['E->E/I'] in ['Allen_V1', 'Allen_custom']:
                prob = '%f * exp(-dist_2D/%f)' % (pmat[pre][post], lmat[pre][post])
            else:
                prob = pmat[pre][post]
            netParams.connParams['EE_'+pre+'_'+post] = { 
                'preConds': {'pop': pre}, 
                'postConds': {'pop': post},
                'synMech': ESynMech,
                'probability': prob,
                'weight': wmat[pre][post] * cfg.EEGain, 
                'synMechWeightFactor': cfg.synWeightFractionEE,
                'delay': 'defaultDelay+dist_3D/propVelocity',
                'synsPerConn': 1,
                'sec': 'dend_all'}
                

#------------------------------------------------------------------------------
## E -> I
if cfg.addConn:
    for pre in Epops:
        for post in Ipops:
            if connDataSource['E->E/I'] in ['Allen_V1', 'Allen_custom']:
                prob = '%f * exp(-dist_2D/%f)' % (pmat[pre][post], lmat[pre][post])
            else:
                prob = pmat[pre][post]
            netParams.connParams['EI_'+pre+'_'+post] = { 
                'preConds': {'pop': pre}, 
                'postConds': {'pop': post},
                'synMech': ESynMech,
                'probability': prob,
                'weight': wmat[pre][post] * cfg.EIGain, 
                'synMechWeightFactor': cfg.synWeightFractionEI,
                'delay': 'defaultDelay+dist_3D/propVelocity',
                'synsPerConn': 1,
                'sec': 'proximal'}
                

#------------------------------------------------------------------------------
## I -> E
if cfg.addConn and cfg.IEGain > 0.0:

    if connDataSource['I->E/I'] == 'Allen_custom':

        ESynMech = ['AMPA', 'NMDA']
        SOMESynMech = ['GABAASlow','GABAB']
        SOMISynMech = ['GABAASlow']
        PVSynMech = ['GABAA']
        VIPSynMech = ['GABAA_VIP']
        NGFSynMech = ['GABAA', 'GABAB']
                        
        layerGroupLabels = ['1-3', '4', '5', '6']

        for pre in Ipops:
            for post in Epops:
                for l in layerGroupLabels:  # used to tune each layer group independently
                    
                    prob = '%f * exp(-dist_2D/%f)' % (pmat[pre][post], lmat[pre][post])
                    
                    if 'SOM' in pre:
                        synMech = SOMESynMech
                    elif 'PV' in pre:
                        synMech = PVSynMech
                    elif 'VIP' in pre:
                        synMech = VIPSynMech
                    elif 'NGF' in pre:
                        synMech = NGFSynMech

                    netParams.connParams['IE_'+pre+'_'+post+'_'+l] = { 
                        'preConds': {'pop': pre}, 
                        'postConds': {'pop': post, 'ynorm': layerGroups[l]},
                        'synMech': synMech,
                        'probability': prob,
                        'weight': wmat[pre][post] * cfg.IEGain * cfg.IELayerGain[l], 
                        'synMechWeightFactor': cfg.synWeightFractionEI,
                        'delay': 'defaultDelay+dist_3D/propVelocity',
                        'synsPerConn': 1,
                        'sec': 'proximal'}
                    

#------------------------------------------------------------------------------
## I -> I
if cfg.addConn and cfg.IIGain > 0.0:

    if connDataSource['I->E/I'] == 'Allen_custom':

        layerGroupLabels = ['1-3', '4', '5', '6']
        for pre in Ipops:
            for post in Ipops:
                for l in layerGroupLabels: 
                    
                    prob = '%f * exp(-dist_2D/%f)' % (pmat[pre][post], lmat[pre][post])

                    if 'SOM' in pre:
                        synMech = SOMISynMech
                    elif 'PV' in pre:
                        synMech = PVSynMech
                    elif 'VIP' in pre:
                        synMech = VIPSynMech
                    elif 'NGF' in pre:
                        synMech = NGFSynMech

                    netParams.connParams['II_'+pre+'_'+post+'_'+l] = { 
                        'preConds': {'pop': pre}, 
                        'postConds': {'pop': post,  'ynorm': layerGroups[l]},
                        'synMech': synMech,
                        'probability': prob,
                        'weight': wmat[pre][post] * cfg.IIGain * cfg.IILayerGain[l], 
                        'synMechWeightFactor': cfg.synWeightFractionII,
                        'delay': 'defaultDelay+dist_3D/propVelocity',
                        'synsPerConn': 1,
                        'sec': 'proximal'}
                        

#------------------------------------------------------------------------------
# Thalamic connectivity parameters
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
## Intrathalamic 

TEpops = ['TC', 'TCM', 'HTC']
TIpops = ['IRE', 'IREM', 'TI']

if cfg.addIntraThalamicConn:
    for pre in TEpops+TIpops:
        for post in TEpops+TIpops:
            if post in pmat[pre]:
                # for syns use ESynMech, SOMESynMech and SOMISynMech 
                if pre in TEpops:     # E->E
                    syn = ESynMech
                    synWeightFactor = cfg.synWeightFractionEE
                elif post in TEpops:  # I->E
                    syn = SOMESynMech
                    synWeightFactor = cfg.synWeightFractionIE
                else:                  # I->I
                    syn = SOMISynMech
                    synWeightFactor = [1.0]
                    
                netParams.connParams['ITh_'+pre+'_'+post] = { 
                    'preConds': {'pop': pre}, 
                    'postConds': {'pop': post},
                    'synMech': syn,
                    'probability': pmat[pre][post],
                    'weight': wmat[pre][post] * cfg.intraThalamicGain, 
                    'synMechWeightFactor': synWeightFactor,
                    'delay': 'defaultDelay+dist_3D/propVelocity',
                    'synsPerConn': 1,
                    'sec': 'soma'}  


#------------------------------------------------------------------------------
## Corticothalamic 
if cfg.addCorticoThalamicConn:
    for pre in Epops:
        for post in TEpops+TIpops:
            if post in pmat[pre]:
                netParams.connParams['CxTh_'+pre+'_'+post] = { 
                    'preConds': {'pop': pre}, 
                    'postConds': {'pop': post},
                    'synMech': ESynMech,
                    'probability': pmat[pre][post],
                    'weight': wmat[pre][post] * cfg.corticoThalamicGain, 
                    'synMechWeightFactor': cfg.synWeightFractionEE,
                    'delay': 'defaultDelay+dist_3D/propVelocity',
                    'synsPerConn': 1,
                    'sec': 'soma'}  

#------------------------------------------------------------------------------
## Thalamocortical 
if cfg.addThalamoCorticalConn:
    for pre in TEpops+TIpops:
        for post in Epops+Ipops:
            if post in pmat[pre]:
                # for syns use ESynMech, SOMESynMech and SOMISynMech 
                if pre in TEpops:     # E->E/I
                    syn = ESynMech
                    synWeightFactor = cfg.synWeightFractionEE
                elif post in Epops:  # I->E
                    syn = SOMESynMech
                    synWeightFactor = cfg.synWeightFractionIE
                else:                  # I->I
                    syn = SOMISynMech
                    synWeightFactor = [1.0]

                netParams.connParams['ThCx_'+pre+'_'+post] = { 
                    'preConds': {'pop': pre}, 
                    'postConds': {'pop': post},
                    'synMech': syn,
                    'probability': '%f * exp(-dist_2D/%f)' % (pmat[pre][post], lmat[pre][post]),
                    'weight': wmat[pre][post] * cfg.thalamoCorticalGain, 
                    'synMechWeightFactor': synWeightFactor,
                    'delay': 'defaultDelay+dist_3D/propVelocity',
                    'synsPerConn': 1,
                    'sec': 'soma'}  


#------------------------------------------------------------------------------
# Subcellular connectivity (synaptic distributions)
#------------------------------------------------------------------------------  

# Set target sections (somatodendritic distribution of synapses)
# From Billeh 2019 (Allen V1) (fig 4F) and Tremblay 2016 (fig 3)

if cfg.addSubConn:
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
        'groupSynMechs': NGFSynMech, 
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    # NGF2,3,4 -> E2,3,4: apic_trunk
    netParams.subConnParams['NGF2,3,4->E2,3,4'] = {
        'preConds': {'pops': ['NGF2', 'NGF3', 'NGF4']}, 
        'postConds': {'pops': ['IT2', 'IT3', 'ITP4', 'ITS4']},
        'sec': 'apic_trunk',
        'groupSynMechs': NGFSynMech, 
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    # NGF2,3,4 -> E5,6: apic_uppertrunk
    netParams.subConnParams['NGF2,3,4->E5,6'] = {
        'preConds': {'pops': ['NGF2', 'NGF3', 'NGF4']}, 
        'postConds': {'pops': ['IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6']},
        'sec': 'apic_uppertrunk',
        'groupSynMechs': NGFSynMech, 
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    # NGF5,6 -> E5,6: apic_lowerrunk
    netParams.subConnParams['NGF5,6->E5,6'] = {
        'preConds': {'pops': ['NGF5A', 'NGF5B', 'NGF6']}, 
        'postConds': {'pops': ['IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6']},
        'sec': 'apic_lowertrunk',
        'groupSynMechs': NGFSynMech, 
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

#------------------------------------------------------------------------------
# Subcellular connectivity (synaptic distributions)
#------------------------------------------------------------------------------  


'''
#------------------------------------------------------------------------------
# Bakcground inputs 
#------------------------------------------------------------------------------  
if cfg.addBkgConn:
    # add bkg sources for E and I cells
    netParams.stimSourceParams['excBkg'] = {'type': 'NetStim', 'start': cfg.startBkg, 'rate': cfg.rateBkg['exc'], 'noise': cfg.noiseBkg, 'number': 1e9}
    netParams.stimSourceParams['inhBkg'] = {'type': 'NetStim', 'start': cfg.startBkg, 'rate': cfg.rateBkg['inh'], 'noise': cfg.noiseBkg, 'number': 1e9}
    
    if cfg.cochlearThalInput:
        from input import cochlearInputSpikes
        numCochlearCells = cfg.cochlearThalInput['numCells']
        cochlearSpkTimes = cochlearInputSpikes(numCells = numCochlearCells,
                                               duration = cfg.duration,
                                               freqRange = cfg.cochlearThalInput['freqRange'],
                                               toneFreq=cfg.cochlearThalInput['toneFreq'],
                                               loudnessDBs=cfg.cochlearThalInput['loudnessDBs'])
                                              
        netParams.popParams['cochlea'] = {'cellModel': 'VecStim', 'numCells': numCochlearCells, 'spkTimes': cochlearSpkTimes, 'ynormRange': layer['cochlear']}

    if cfg.ICThalInput:
        # load file with IC output rates
        from scipy.io import loadmat
        import numpy as np

        data = loadmat(cfg.ICThalInput['file'])
        fs = data['RsFs'][0][0]
        ICrates = data['BE_sout_population'].tolist()
        ICtimes = list(np.arange(0, cfg.duration, 1000./fs))  # list with times to set each time-dep rate
        
        
        ICrates = ICrates * 4 # 200 cells
        
        numCells = len(ICrates)

        # Option 1: create population of DynamicNetStims with time-varying rates
        #netParams.popParams['IC'] = {'cellModel': 'DynamicNetStim', 'numCells': numCells, 'ynormRange': layer['cochlear'],
        #    'dynamicRates': {'rates': ICrates, 'times': ICtimes}}

        # Option 2:
        from input import inh_poisson_generator
        
        maxLen = min(len(ICrates[0]), len(ICtimes))
        spkTimes = [[x+cfg.ICThalInput['startTime'] for x in inh_poisson_generator(ICrates[i][:maxLen], ICtimes[:maxLen], cfg.duration, cfgICThalInput['seed'])] for i in range(len(ICrates))]
        netParams.popParams['IC'] = {'cellModel': 'VecStim', 'numCells': numCells, 'ynormRange': layer['cochlear'],
            'spkTimes': spkTimes}



    # excBkg/I -> thalamus + cortex
    with open('cells/bkgWeightPops.json', 'r') as f:
        weightBkg = json.load(f)
    pops = list(cfg.allpops)
    pops.remove('IC')
    for pop in pops:
        netParams.stimTargetParams['excBkg->'+pop] =  {
            'source': 'excBkg', 
            'conds': {'pop': pop},
            'sec': 'apic', 
            'loc': 0.5,
            'synMech': ESynMech,
            'weight': weightBkg[pop]*10,
            'synMechWeightFactor': cfg.synWeightFractionEE,
            'delay': cfg.delayBkg}

        netParams.stimTargetParams['inhBkg->'+pop] =  {
            'source': 'inhBkg', 
            'conds': {'pop': pop},
            'sec': 'proximal',
            'loc': 0.5,
            'synMech': 'GABAA',
            'weight': weightBkg[pop]*2,
            'delay': cfg.delayBkg}


# ------------------------------------------------------------------------------
# Current inputs (IClamp)
# ------------------------------------------------------------------------------
if cfg.addIClamp:
    for key in [k for k in dir(cfg) if k.startswith('IClamp')]:
        params = getattr(cfg, key, None)
        [pop,sec,loc,start,dur,amp] = [params[s] for s in ['pop','sec','loc','start','dur','amp']]
        
        if cfg.singlePop:
            pop = cfg.singlePop        
        
        # add stim source
        netParams.stimSourceParams[key] = {'type': 'IClamp', 'delay': start, 'dur': dur, 'amp': amp}
        
        # connect stim source to target
        netParams.stimTargetParams[key+'_'+pop] =  {
            'source': key, 
            'conds': {'pop': pop},
            'sec': sec, 
            'loc': loc}

#------------------------------------------------------------------------------
# NetStim inputs (to simulate short external stimuli; not bkg)
#------------------------------------------------------------------------------
if cfg.addNetStim:
    for key in [k for k in dir(cfg) if k.startswith('NetStim')]:
        params = getattr(cfg, key, None)
        [pop, ynorm, sec, loc, synMech, synMechWeightFactor, start, interval, noise, number, weight, delay] = \
        [params[s] for s in ['pop', 'ynorm', 'sec', 'loc', 'synMech', 'synMechWeightFactor', 'start', 'interval', 'noise', 'number', 'weight', 'delay']]

        cfg.analysis['plotTraces']['include'] = [(pop, 0)]

        # add stim source
        netParams.stimSourceParams[key] = {'type': 'NetStim', 'start': start, 'interval': interval, 'noise': noise, 'number': number}

        # connect stim source to target 
        netParams.stimTargetParams[key+'_'+pop] =  {
            'source': key, 
            'conds': {'pop': pop, 'ynorm': ynorm},
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
v25- Fixed subconnparams TC->E and NGF1->E; made IC input deterministic
"""