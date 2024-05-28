"""
script to load sim and plot
"""

from matplotlib import pyplot as plt
import os, collections

import IPython as ipy
import pickle as pkl
import json
import numpy as np
import scipy.signal as ss
from netpyne import specs,sim

from shared import *  ## 


# with open('../../sim/cells/popColors.pkl', 'rb') as fileObj: popColors = pkl.load(fileObj)['popColors']
with open('popColors.pkl', 'rb') as fileObj: popColors = pkl.load(fileObj)['popColors']

print('testing vscode change')

def loadFile(filename, include):

    if filename.endswith('.json'):
        with open(filename, 'rb') as fileObj:
            data = json.load(fileObj, object_pairs_hook=collections.OrderedDict)
    elif filename.endswith('.pkl'):
        with open(filename, 'rb') as fileObj:
            data = pkl.load(fileObj)

    cfg = specs.SimConfig(data['simConfig'])
    cfg.createNEURONObj = False

    sim.initialize()  # create network object and set cfg and net params
    sim.loadAll('', data=data, instantiate=False)
    sim.setSimCfg(cfg)
    
    temp = collections.OrderedDict()

    # order pops by keys in popColors
    for k in include:
        if k in sim.net.params.popParams:
            temp[k] = sim.net.params.popParams.pop(k)

    # add remaining pops at the end
    for k in list(sim.net.params.popParams.keys()):
        temp[k] = sim.net.params.popParams.pop(k)
    
    sim.net.params.popParams = temp
    
    try:
        print('Cells created: ',len(sim.net.allCells))
    except:
        #import IPython; IPython.embed()
        sim.net.createPops()     
        sim.net.createCells()
        sim.setupRecording()
        sim.cfg.createNEURONObj = False
        sim.gatherData() 

    sim.allSimData = data['simData']



# ###########################
# ######## MAIN CODE ########
# ###########################

if __name__ == '__main__':

    # ## Osc Event Info files ##
    # oscEventDir = '../data/topPops/AllCortPops/'            # Could be ECortPops or ICortPops
    # oscEventFiles = []
    # filesOscEventDir = os.listdir(oscEventDir)
    # for file in filesOscEventDir:
    #     if '.pkl' in file:
    #         oscEventFiles.append(file)



    # ## PICK A FREQUENCY BAND -- iterate over later 
    # frequencyBand = ['delta'] # ['alpha', 'beta', 'theta']

    # ## PICK A REGION -- iterate over later 
    # corticalRegion = ['supra'] # ['infra', 'gran']


    # for band in frequencyBand:
    #     ## load osc event info pkl file --> e.g. here it will load delta osc event pkl file
    #     #### NOTE: Fix this during iteration process --- for now designed just to look at one particular file and go from there
    #     for file in oscEventFiles:
    #         if frequencyBand in file:
    #             with open(file, 'rb') as handle:
    #                 oscEventData = pickle.load(file)
    #     ## Now 'delta' osc event info file should be loaded (from data/topPops/AllCortPops/)

    #     ## Now get a list of osc event indices in the band x region specified (e.g. delta, supra)
    #     oscEventIdx = []
    #     A1_subjects = []
    #     for region in corticalRegion:
    #         A1_subjects = list(oscEventData[band][region].keys())





    ## Get all data dir files ##
    dataDir = '../data/simDataFiles/spont/v34_batch67_CINECA/data_pklFiles/'
    filenames = []
    filesDataDir = os.listdir(dataDir)
    for file in filesDataDir:
        if '.pkl' in file:
            filenames.append(file)
    filenames.sort()

    ## List for testing specific files ##
    #filenames = ['../data/simDataFiles/spont/v34_batch67_CINECA/data_pklFiles/v34_batch67_CINECA_0_0_data.pkl']
    filenames = ['../data/simDataFiles/BBN/BBN_CINECA_v36_5656BF_SOA850/BBN_CINECA_v36_5656BF_850ms_data.pkl', '../data/simDataFiles/spont/v34_batch67_CINECA/data_pklFiles/v34_batch67_CINECA_0_0_data.pkl']


    timeRanges = [[0,11500], [0,11500]]    #[[1000,5000], [5000, 9000]]
    freqRanges = [[1,80], [1,80]] #[[0,4], [30,80]]

    individualPlots = False # True
    combinedPlots = True # False


    allpops = ['NGF1', 
                    'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 
                    'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 
                    'ITP4', 'ITS4', 'PV4', 'SOM4', 'VIP4', 'NGF4', 
                    'IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A', 
                    'IT5B', 'PT5B', 'CT5B', 'PV5B', 'SOM5B', 'VIP5B', 'NGF5B', 
                    'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6']

    excpops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'CT5B', 'PT5B', 'IT6', 'CT6']


    ## SOLUTION TO POPCOLORS ERROR for COLORS in DATA PLOTTING!! 
    colorList = [[0.42,0.67,0.84], [0.90,0.76,0.00], [0.42,0.83,0.59], [0.90,0.32,0.00],
                [0.34,0.67,0.67], [0.90,0.59,0.00], [0.42,0.82,0.83], [1.00,0.85,0.00],
                [0.33,0.67,0.47], [1.00,0.38,0.60], [0.57,0.67,0.33], [0.5,0.2,0.0],
                [0.71,0.82,0.41], [0.0,0.2,0.5], [0.70,0.32,0.10]]*3
    popColors = {}
    for p in range(len(allpops)):
        popColors[allpops[p]] = colorList[p]
    ####### 

    loadProcessedData = 0 #1 

    fontsiz = 16

    for filename, timeRange, freqRange in zip(filenames, timeRanges, freqRanges):
        print('filename: ' + str(filename))
        print('timeRange: ' + str(timeRange))

        if not loadProcessedData:
            loadFile(filename, include=allpops)

            allDataLFP = []
            allDataPSD = []

            # # raster   ## NOTE: THIS IS COMMENTED OUT IN M1_PSD.py
            # sim.analysis.plotRaster(
            #     include = allpops, 
            #     popColors = popColors,
            #     popRates = 'minimal', 
            #     orderInverse = True, 
            #     timeRange = timeRange,
            #     orderBy = ['pop','y'], 
            #     figSize = (12,8), 
            #     lw = 0.3, 
            #     markerSize = 3, 
            #     marker = '.', 
            #     dpi = 600,
            #     saveFig = True, 
            #     showFig = False)

            # LFP 
            if individualPlots:
                # Overall
                out = sim.plotting.plotLFPTimeSeries( 
                        electrodes = [6], #['avg']+list(range(11)), 
                        timeRange = timeRange, 
                        figSize = (16,12), 
                        rcParams = {'font.size': 20},
                        saveData = False, 
                        saveFig = True,
                        fileName = filename[ :-4]+'_LFP_timeSeries_allpops_%d_%d.png'%(timeRange[0], timeRange[1]), 
                        showFig = False)


                out = sim.plotting.plotLFPPSD(
                        plots = ['PSD'], 
                        electrodes = [6],
                        timeRange = timeRange, 
                        minFreq = 0.05,
                        maxFreq = 80,
                        stepFreq = 0.05,
                        orderInverse = False, 
                        figSize = (8,6), 
                        rcParams = {'font.size': 20},
                        saveData = False, 
                        saveFig = True,
                        fileName = filename[:-4]+'_LFP_PSD_allpops_%d_%d.png'%(timeRange[0], timeRange[1]), 
                        showFig = False)


                # By population
                for pop in allpops:   #['IT2','IT4']: #,'IT5A','IT5B','PT5B','SOM5B','PV5B','IT6','CT6']:
                    out = sim.plotting.plotLFPTimeSeries( 
                            electrodes = [6], #['avg']+list(range(11)), 
                            pop = pop,
                            timeRange = timeRange, 
                            figSize = (16,12), 

                            rcParams = {'font.size': 20},
                            saveData = False, 
                            saveFig = True,
                            fileName = filename[ :-4]+'_LFP_timeSeries_%s_%d_%d.png'%(pop, timeRange[0], timeRange[1]), 
                            showFig = False)
                    
                    out = sim.plotting.plotLFPPSD(
                            plots = ['PSD'], 
                            electrodes = [6],   # NOTE THAT ELECTRODE WILL HAVE TO CORRESPOND TO ELEC OF OSC EVENT! 
                            pop = pop,
                            timeRange = timeRange, 
                            minFreq = 0.05,
                            maxFreq = 80,
                            stepFreq = 0.05,
                            orderInverse = False, 
                            figSize = (8,6), 
                            rcParams = {'font.size': 20},
                            saveData = False, 
                            saveFig = True,
                            fileName = filename[:-4]+'_LFP_PSD_%s_%d_%d.png'%(pop, timeRange[0], timeRange[1]), 
                            showFig = False)

            if combinedPlots:
                ## ELECTRODE --> PUT THIS SOMEWHERE ELSE!! 
                elec = [13] #[10] #['avg']
                # Overall
                dataLFP = sim.analysis.prepareLFP( 
                        electrodes = elec, #[6], #['avg']+list(range(11)), 
                        timeRange = timeRange, 
                        filtFreq = 200,
                        figSize = (16,12), 
                        rcParams = {'font.size': 20},
                        saveData = False, 
                        saveFig = True,
                        fileName = filename[ :-4]+'_LFP_timeSeries_allpops_%d_%d.png'%(timeRange[0], timeRange[1]), 
                        showFig = False)

                dataPSD = sim.analysis.preparePSD(
                        electrodes = elec, #[6],
                        timeRange = timeRange, 
                        minFreq = 0.05,
                        maxFreq = 80,
                        stepFreq = 0.05,
                        orderInverse = False, 
                        figSize = (8,6), 
                        rcParams = {'font.size': 20},
                        saveData = False, 
                        saveFig = True,
                        fileName = filename[:-4]+'_LFP_PSD_allpops_%d_%d.png'%(timeRange[0], timeRange[1]), 
                        showFig = False)

                allDataLFP.append(dataLFP)
                allDataPSD.append(dataPSD)


                # By population
                ### TESTING LINES --> EYG 12/06 --> instead of allpops --> testpops = ['IT2', 'IT5A'] 
                testpops = [] #['IT5A', 'CT6']  #['IT2', 'IT5A', 'CT6'] 
                for pop in testpops: #allpops:
                    print('pop: ' + pop)
                    dataLFP = sim.analysis.prepareLFP( 
                            electrodes = elec, #[6], #['avg']+list(range(11)), 
                            pop = pop,
                            timeRange = timeRange, 
                            filtFreq = 200,
                            figSize = (16,12), 
                            rcParams = {'font.size': 20},
                            saveData = False, 
                            saveFig = True,
                            fileName = filename[ :-4]+'_LFP_timeSeries_%s_%d_%d.png'%(pop, timeRange[0], timeRange[1]), 
                            showFig = False)
                    
                    dataPSD = sim.analysis.preparePSD(
                            plots = ['PSD'], 
                            electrodes = elec, #[6],
                            pop = pop,
                            timeRange = timeRange, 
                            minFreq = 0.05,
                            maxFreq = 80,
                            stepFreq = 0.05,
                            orderInverse = False, 
                            figSize = (8,6), 
                            rcParams = {'font.size': 20},
                            saveData = False, 
                            saveFig = True,
                            fileName = filename[:-4]+'_LFP_PSD_%s_%d_%d.png'%(pop, timeRange[0], timeRange[1]), 
                            showFig = False)

                    ## NOTE: these dicts (at least dataPSD) do not appear to have the pop names associated with them? hm. 
                    allDataLFP.append(dataLFP) 
                    allDataPSD.append(dataPSD)
            
            # save processed data to file
            with open(filename[ :-4]+'_processed_data.pkl', 'wb') as f:
                pickle.dump([allDataLFP, allDataPSD] ,f)

        # load processed data from file
        else:
            with open(filename[ :-4]+'_processed_data.pkl', 'rb') as f:
                [allDataLFP, allDataPSD] = pickle.load(f)

        # calculate pops contributing most to psd
    #freqRange = [x*200 for x in freqRanges[]] #[0, 1600]  # 0-80 hz indices  #for 25 to 50 hz
    freqScale = 20
    #### freqPeaks is the line where "frequency power is calculated based on psd"!! ####
    freqPeaks = [np.sum(x['psdSignal'][0][freqRange[0]*freqScale:freqRange[1]*freqScale]) for x in allDataPSD]
    topPopIndices = np.argsort(freqPeaks)[::-1]
    topPopIndices = topPopIndices[:5]

    # calculate pops contributing most to lfp amplitude
    # peaks = [np.max(np.mean(np.abs(x['electrodes']['lfps'][0]))) for x in allDataLFP]
    # topPopIndices = np.argsort(peaks)[::-1]
    # topPopIndices = topPopIndices[:6]

    # plotting
    popLabels = ['All']+testpops #allpops
    popColors['All'] = 'black'

    plt.figure(figsize=(8,4))
    plt.ion()
    fs = 1000/0.025  

    for i in topPopIndices:
        # combined PSD 
        # using netpyne PSD
        dataNorm = allDataPSD[i]['psdSignal'][0] #/ np.max(allDataPSD[0]['psdSignal'][0])
        f = allDataPSD[i]['psdFreqs'][0]
        plt.plot(f, dataNorm*1000, label=popLabels[i],  color=popColors[popLabels[i]], linewidth=2) 
        print('popLabel: ' + str(popLabels[i]))
        ## ^^ NOTE: can I verify that the label corresponds to whatever is being plotted...? 

        # using Welch
        # x = allDataLFP[i]['electrodes']['lfps'][0]
        # f, Pxx_den = ss.welch(x, fs, nperseg=len(x)/2)
        # if popLabels[i] == 'All':
        #     plt.semilogy(f, Pxx_den*1000, label=popLabels[i], color=popColors[popLabels[i]], linewidth=3)#, linestyle=':') # in uV
        # else:
        #     plt.semilogy(f, Pxx_den*1000, label=popLabels[i], color=popColors[popLabels[i]], linewidth=2 ) # in uV
        # if periodLabel == 'quiet':
        #     plt.ylim([5e-6,3])
        # elif periodLabel == 'move':
        #     plt.ylim([0.3e-4,1e-0])
        # plt.xlim([0.0, 80])
        # plt.grid(False)

    # log x and log y (for wavelet/netpyne PSD)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlim(1,80) 
    plt.ylim([.4e-3,4e0]) #([.4e-3,4e0])

    #ipy.embed()

    xstep = 20
    #xrange = np.arange(xstep, np.max(f)+1, xstep)
    xticks = [1, 4, 8, 12, 30, 80]# [3, 9, 28, 80]
    ax.set_xticks(xticks)
    ax.set_xticklabels(['%.0f' % x for x in xticks])

    # log freq
    # ax.set_xscale('log')
    # plt.xlim([0.0, 80])

    ax = plt.gca()
    plt.setp(ax.get_xticklabels(),fontsize=fontsiz)
    plt.setp(ax.get_yticklabels(),fontsize=fontsiz)
    plt.xlabel('Frequency (Hz)', fontsize=fontsiz)
    plt.ylabel('LFP power ($\mu$V$^2$/Hz)', fontsize=fontsiz) #'PSD [V**2/Hz]'
    plt.legend(fontsize=fontsiz, loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.subplots_adjust(bottom=0.15, top=0.95 , left=0.1, right=0.82 )

    plt.show()
    # plt.savefig(filename[:-4]+'_LFP_PSD_%s_combined_Welch.png' % (periodLabel), dpi=300)
    plt.savefig(filename[:-4]+'_LFP_PSD_combined_Welch.png', dpi=300)
    ### ^^ NOTE: WHY DOES THIS HAVE COMBINED WELCH IN IT WHEN IT APPEARS THAT THE WELCH LINES ABOVE ARE COMMENTED OUT? 


        # # calculate pops contributing most to psd
        # #freqRange = [x*200 for x in freqRanges[]] #[0, 1600]  # 0-80 hz indices  #for 25 to 50 hz
        # freqScale = 20
        # #### freqPeaks is the line where "frequency power is calculated based on psd"!! ####
        # freqPeaks = [np.sum(x['psdSignal'][0][freqRange[0]*freqScale:freqRange[1]*freqScale]) for x in allDataPSD]
        # topPopIndices = np.argsort(freqPeaks)[::-1]
        # topPopIndices = topPopIndices[:5]

        # # calculate pops contributing most to lfp amplitude
        # # peaks = [np.max(np.mean(np.abs(x['electrodes']['lfps'][0]))) for x in allDataLFP]
        # # topPopIndices = np.argsort(peaks)[::-1]
        # # topPopIndices = topPopIndices[:6]

        # # plotting
        # popLabels = ['All']+testpops #allpops
        # popColors['All'] = 'black'

        # plt.figure(figsize=(8,4))
        # plt.ion()
        # fs = 1000/0.025  

        # for i in topPopIndices:
        #     # combined PSD 
        #     # using netpyne PSD
        #     dataNorm = allDataPSD[i]['psdSignal'][0] #/ np.max(allDataPSD[0]['psdSignal'][0])
        #     f = allDataPSD[i]['psdFreqs'][0]
        #     plt.plot(f, dataNorm*1000, label=popLabels[i],  color=popColors[popLabels[i]], linewidth=2) 
        #     print('popLabel: ' + str(popLabels[i]))
        #     ## ^^ NOTE: can I verify that the label corresponds to whatever is being plotted...? 

        #     # using Welch
        #     # x = allDataLFP[i]['electrodes']['lfps'][0]
        #     # f, Pxx_den = ss.welch(x, fs, nperseg=len(x)/2)
        #     # if popLabels[i] == 'All':
        #     #     plt.semilogy(f, Pxx_den*1000, label=popLabels[i], color=popColors[popLabels[i]], linewidth=3)#, linestyle=':') # in uV
        #     # else:
        #     #     plt.semilogy(f, Pxx_den*1000, label=popLabels[i], color=popColors[popLabels[i]], linewidth=2 ) # in uV
        #     # if periodLabel == 'quiet':
        #     #     plt.ylim([5e-6,3])
        #     # elif periodLabel == 'move':
        #     #     plt.ylim([0.3e-4,1e-0])
        #     # plt.xlim([0.0, 80])
        #     # plt.grid(False)

        # # log x and log y (for wavelet/netpyne PSD)
        # ax = plt.gca()
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        # plt.xlim(1,80) 
        # plt.ylim([.4e-3,4e0]) #([.4e-3,4e0])

        # #ipy.embed()

        # xstep = 20
        # xrange = np.arange(xstep, np.max(f)+1, xstep)
        # xticks = [1, 4, 8, 12, 30, 80]# [3, 9, 28, 80]
        # ax.set_xticks(xticks)
        # ax.set_xticklabels(['%.0f' % x for x in xticks])

        # # log freq
        # # ax.set_xscale('log')
        # # plt.xlim([0.0, 80])

        # ax = plt.gca()
        # plt.setp(ax.get_xticklabels(),fontsize=fontsiz)
        # plt.setp(ax.get_yticklabels(),fontsize=fontsiz)
        # plt.xlabel('Frequency (Hz)', fontsize=fontsiz)
        # plt.ylabel('LFP power ($\mu$V$^2$/Hz)', fontsize=fontsiz) #'PSD [V**2/Hz]'
        # plt.legend(fontsize=fontsiz, loc='upper left', bbox_to_anchor=(1.01, 1))
        # plt.subplots_adjust(bottom=0.15, top=0.95 , left=0.1, right=0.82 )

        # plt.show()
        # # plt.savefig(filename[:-4]+'_LFP_PSD_%s_combined_Welch.png' % (periodLabel), dpi=300)
        # plt.savefig(filename[:-4]+'_LFP_PSD_combined_Welch.png', dpi=300)
        # ### ^^ NOTE: WHY DOES THIS HAVE COMBINED WELCH IN IT WHEN IT APPEARS THAT THE WELCH LINES ABOVE ARE COMMENTED OUT? 


        # plotEachLFP = 0 # 0 

        # if plotEachLFP:

        #     for i in topPopIndices:
        #         # individual LFP timeseries
        #         plt.figure(figsize=(8,4))
        #         lw=0.5
        #         t = allDataLFP[i]['t']
        #         plt.plot(t, -allDataLFP[i]['electrodes']['lfps'][0], color=popColors[popLabels[i]], linewidth=lw)
        #         ax = plt.gca()        
        #         ax.invert_yaxis()
        #         plt.axis('off')
        #         plt.xlabel('time (ms)', fontsize=fontsiz)

        #         meanSignal = np.mean(-allDataLFP[i]['electrodes']['lfps'][0])
        #         plt.ylim(meanSignal+0.4,meanSignal-0.5)
        #         plt.subplots_adjust(bottom=0.0, top=0.9, left=0.1, right=0.9)

        #         # calculate scalebar size and add scalebar
        #         scaley = 1000.0  # values in mV but want to convert to uV
        #         sizey = 100/scaley
        #         labely = '%.3g $\mu$V'%(sizey*scaley)#)[1:]
        #         sizex=500
        #         add_scalebar(ax, hidey=True, matchy=False, hidex=True, matchx=True, sizex=None, sizey=-sizey, labely=labely, unitsy='$\mu$V', scaley=scaley, 
        #                 unitsx='ms', loc='upper right', pad=0.5, borderpad=0.5, sep=3, prop=None, barcolor="black", barwidth=2)
                
        #         plt.title('LFP 0-200 Hz', fontsize=fontsiz, fontweight='bold')
        #         # plt.savefig(filename[:-4]+'_LFP_timeSignal_%s_%s.png' % (popLabels[i], periodLabel),dpi=300)
        #         plt.savefig(filename[:-4]+'_LFP_timeSignal_%s.png' % popLabels[i], dpi=300)

            
        
#         #ipy.embed()
