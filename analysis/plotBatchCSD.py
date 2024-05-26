"""
script to load sim and plot
"""

from netpyne import sim
from matplotlib import pyplot as plt
import os
import IPython as ipy
import scipy.io 
import numpy as np

targetFolder = '../data/simDataFiles/spont/'   #'../data/NHPdata/spont/contproc/A1'  #'data/v34_batch27'
saveFolderName = '../data/figs/spontSimCSD/newTestCSD/ '#2000ms_csd_window/'

# get all the filenames 
filenames = ['A1_v34_batch67_v34_batch67_0_0_data.pkl']#, 'A1_v34_batch67_v34_batch67_1_1_data.pkl', 'A1_v34_batch65_v34_batch65_0_0_data.pkl', 
# 2-bu027028013@os_eye06_20
# 2-bu043044016@os_eye06_20
# 2-gt044045014@os_eye06_30
# 2-ma031032023@os_eye06_20
# 2-rb031032016@os_eye06_20
# 2-rb045046026@os_eye06_20
# 2-rb063064011@os_eye06_20

# dbpath = 21feb02_A1_spont_layers.csv
## will have to be inside the for loop 
## will need fx getflayers 
getFigs = 1
getData = 0


layer_bounds= {'L1': 100, 'L2': 160, 'L3': 950, 'L4': 1250, 'L5A': 1334, 'L5B': 1550, 'L6': 2000}



if getFigs:
    for filename in filenames:
        fullPathFilename = targetFolder + filename
        saveFigName = saveFolderName + filename[:-4]

        sim.load(fullPathFilename, instantiate=False)

        tranges = [[x, x+2000] for x in range(500, 10500, 2000)]

        for t in tranges:
            sim.analysis.plotCSD(**{
                'spacing_um': 100,  
                'timeRange': [t[0], t[1]], 
                'smooth': 30,
                'saveFig': saveFigName+'_CSD_%d-%d' % (t[0], t[1]),
                'figSize': (6,9), 
                'dpi': 300, 
                'showFig': 0})      #'layer_lines': 1, 'layer_bounds': layer_bounds, 

###################
# 200 ms examples: 
## batch65_0_0 timeRanges:
### 6900-7100
### 7700-7900
### 4500-4700
# timeRanges = [[6900,7100], [7700,7900], [4500,4700]]

## batch67_0_0 timeRanges:
# timeRanges = [[8100, 8300]]

###################
# 2000 ms examples:

## batch67_0_0
### [6500, 8500]
### [2500, 4500]
### timeRanges = [[2500, 4500],[6500, 8500]]

## batch67_1_1
### [8500,10500]
### [500, 2500]
### timeRanges = [[500, 2500], [8500,10500]]

## batch65_0_0
### [4500, 6500]
timeRanges = [[4500, 6500]]




#filenames = ['A1_v34_batch67_v34_batch67_1_1_data.pkl'] #, 'A1_v34_batch67_v34_batch67_1_1_data.pkl', 'A1_v34_batch67_v34_batch67_0_0_data.pkl', 

filename = 'A1_v34_batch65_v34_batch65_0_0_data.pkl' #, 'A1_v34_batch67_v34_batch67_1_1_data.pkl', 'A1_v34_batch67_v34_batch67_0_0_data.pkl', 

if getData:
    i=0
    for timeRange in timeRanges:                # timeRange = timeRanges[1]      # SET TIME RANGE 
        fullPathFilename = targetFolder + filename

        subjectName = filename.split('_data.pkl')[0]

        sim.load(fullPathFilename, instantiate=False)
        ## LFP 
        lfpDataFull = sim.allSimData['LFP']
        lfpData_timeRange = np.array(sim.allSimData['LFP'])[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]
        
        saveLFPName = '../data/figs/spontSimCSD/' + subjectName + '_timeRange_' + str(timeRange[0]) + '_' + str(timeRange[1]) + '_LFP.mat'
        scipy.io.savemat(saveLFPName, {'LFP_data_timeRange': lfpData_timeRange})

        if i==0:
            print('saving full lfp')
            saveLFPFullName = '../data/figs/spontSimCSD/' + subjectName + '_LFP_FULL.mat'
            scipy.io.savemat(saveLFPFullName, {'LFP_data_FULL': lfpDataFull})
    i+=1 

        ## CSD
        # CSD_data_getCSD_FULL = sim.analysis.getCSD(save_to_sim=True) # Then take timeRange yourself 
        # CSD_data_getCSD = CSD_data_getCSD_FULL[:,int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep)]  

        # saveMatName = '../data/simDataFiles/A1_v34_batch65_0_0_timeRange_' + str(timeRange[0]) + '_' + str(timeRange[1]) + '.mat'
        # scipy.io.savemat(saveMatName, {'CSD_data': CSD_data_getCSD})
        # sim.analysis.plotCSD(timeRange=timeRange, saveFig=False, showFig=False)           # timeRange = []
        # CSD_data_plotCSD = sim.allSimData['CSD']['CSD_data']


