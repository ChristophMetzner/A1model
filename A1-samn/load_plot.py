"""
script to load sim and plot
"""

from netpyne import sim
from matplotlib import pyplot as plt
import os
import IPython as ipy
import pickle as pkl
import numpy as np


def plot_LFP_PSD_combined(dataFile, norm=False):
    NHP = dataFile[:-4]
    
    with open(dataFile, 'rb') as f:
        loadedData = pkl.load(f)
        allData = loadedData['allData']
        freqs = loadedData['allFreqs'][0]

    plt.figure(figsize=(8,12))
    fontSize = 12
    lw = 1

    elecLabels = ['All layers', 'Supragranular', 'Granular', 'Infragranular']

    for i in range(len(elecLabels)):
        plt.subplot(4, 1, i+1)
        for itime in range(len(allData)):
            signal = allData[itime][i]
            if norm:
                signal = signal / max(signal)
            plt.plot(freqs, signal, linewidth=lw) #, color=color)
        
        plt.title(elecLabels[i])
        plt.xlim([0, 50])
        if norm:
            plt.ylabel('Normalized Power', fontsize=fontSize)
        else:
            plt.ylabel('Power (mV^2/Hz)', fontsize=fontSize)
        plt.xlabel('Frequency (Hz)', fontsize=fontSize)
    
    # format plot 
    plt.tight_layout()
    plt.suptitle('LFP PSD - %s' % (NHP), fontsize=fontSize, fontweight='bold') # add yaxis in opposite side
    plt.subplots_adjust(bottom=0.08, top=0.92)

    if norm:
        plt.savefig('%s_PSD_combined_norm.png' % (NHP))
    else:
        plt.savefig('%s_PSD_combined.png' % (NHP))


###########################
######## MAIN CODE ########
###########################

if __name__ == '__main__':

    dataType = 'spont' #'speech' #'spont'

    if dataType == 'spont':
        filenames = ['data/v35_batch5/v35_batch5_%d_%d_data.pkl' % (iseed, cseed) for iseed in [0,1] for cseed in [0,1]]
        #filenames = ['data/v34_batch68/v34_batch68_%d_%d_%d_data.pkl' % (iseed1, iseed2, iseed3) for iseed1 in [0,1,2,3] for iseed2 in [0,1] for iseed3 in [0,1]]
        #filenames = ['data/v34_batch66/v34_batch66_0_0_data.pkl']

        timeRange = [1500, 11450]

        #filenames = ['data/v34_batch56/v34_batch56_0_0_data.pkl']
        #timeRange = [2000, 2200]

        # find all individual sim labels whose files need to be gathered
    #filenames = [targetFolder+f for f in os.listdir(targetFolder) if f.endswith('.pkl')]

    elif dataType == 'speech':
        filenames = ['data/v34_batch62/v34_batch62_%d_%d_%d_%d_%d_%d_data.pkl' % (i1, i1, i1, i1, i2, i3) 
                                                                        for i1 in [1] \
                                                                        for i2 in [1, 2, 3] \
                                                                        for i3 in [0, 1, 2, 3] ]

        # good_rasters = [
        # '0_0_0_0_1',
        # '0_0_0_0_2',
        # ]

        # filenames = ['data/v34_batch55/v34_batch55_%s_data.pkl' % (s1) for s1 in good_rasters] 
        # timeRange = [2500, 4000]

    layer_bounds= {'L1': 100, 'L2': 160, 'L3': 950, 'L4': 1250, 'L5A': 1334, 'L5B': 1550}#, 'L6': 2000}
    layer_bounds= {'S': 950, 'G': 1250, 'I': 1900}#, 'L6': 2000}


    allDataLFP = []
    allDataPSD = []

    for filename in filenames:
        sim.load(filename, instantiate=False)

        # standardd plots
        #sim.analysis.plotRaster(**{'include': ['allCells'], 'saveFig': True, 'showFig': False, 'popRates': 'minimal', 'orderInverse': True, 'timeRange': timeRange, 'figSize': (18,12), 'lw': 0.3, 'markerSize': 3, 'marker': '.', 'dpi': 300})
        # sim.analysis.plotSpikeHist(include=['allCells', 
        #                                     ['IT3', 'PV3', 'SOM3', 'NGF3', 'VIP3'], 
        #                                     ['ITP4','ITS4', 'PV4', 'SOM4', 'NGF4', 'VIP4'], 
        #                                     ['IT5A', 'PV5A', 'SOM5A', 'NGF5A', 'VIP5A'],
        #                                     ['IT5B','PT5B', 'PV5B', 'SOM5B', 'NGF5B', 'VIP5B']], 
        #                             saveFig=1, 
        #                             binSize=1000,
        #                             measure='rate',
        #                             timeRange=timeRange)
        # 
        #sim.analysis.plotSpikeStats(stats=['rate'],figSize = (6,12), timeRange=timeRange, dpi=300, showFig=0, saveFig=filename[:-4]+'_stats_5sec')
        #sim.analysis.plotSpikeStats(stats=['rate'],figSize = (6,12), timeRange=[1500, 6500], dpi=300, showFig=0, saveFig=filename[:-4]+'_stats_5sec')
        #sim.analysis.plotLFP(**{'plots': ['spectrogram'], 'electrodes': ['avg', [0], [1], [2,3,4,5,6,7,8,9], [10, 11, 12], [13], [14, 15], [16,17,18,19]], 'timeRange': timeRange, 'maxFreq': 50, 'figSize': (8,24), 'saveData': False, 'saveFig': filename[:-4]+'_LFP_spec_7s_all_elecs', 'showFig': False})

        for elec in [10,11,12]:
            lfp=np.array(sim.allSimData['LFP'])
            minpos=np.argmin(lfp[3000: , elec])+3000
            print(elec,minpos)
            print(sim.allSimData['LFP'][minpos][elec])
            sim.allSimData['LFP'][minpos][elec] = sim.allSimData['LFP'][minpos-5][elec]
            print(sim.allSimData['LFP'][minpos][elec])

        # for elec in [10,11,12]:
        #     lfp=np.array(sim.allSimData['LFP'])
        #     minpos=np.argmin(lfp[3000: , elec])+3000
        #     print(elec,minpos)
        #     print(sim.allSimData['LFP'][minpos][elec])
        #     sim.allSimData['LFP'][minpos][elec] = sim.allSimData['LFP'][minpos-5][elec]
        #     print(sim.allSimData['LFP'][minpos][elec])

        # for elec in [10,11,12]:
        #     lfp=np.array(sim.allSimData['LFP'])
        #     minpos=np.argmin(lfp[3000: , elec])+3000
        #     print(elec,minpos)
        #     print(sim.allSimData['LFP'][minpos][elec])
        #     sim.allSimData['LFP'][minpos][elec] = sim.allSimData['LFP'][minpos-5][elec]
        #     print(sim.allSimData['LFP'][minpos][elec])


        # out = sim.plotting.plotLFPTimeSeries(**{ 
        #         'electrodes': 
        #         #['avg', 11, 15], 
        #         #['avg', [0, 1,2,3,4,5,6,7,8,9], [10, 11, 12], [13,14, 15,16,17,18,19]],
        #         [[10, 11, 12]],
        #         'timeRange': timeRange, 
        #         'figSize': (8,6), 
        #         'rcParams': {'font.size': 20},
        #         'saveData': False, 
        #         'saveFig': filename[:-4]+'_LFP_timeSeries_%d_%d'%(timeRange[0], timeRange[1]), 
        #         'showFig': False})

        # out = sim.plotting.plotLFPSpectrogram(**{'plots': ['spectrogram'], 
        #         'electrodes': 
        #         #['avg', [0], [1], [2,3,4,5,6,7,8,9], [10, 11, 12], [13], [14, 15], [16,17,18,19]], 
        #         # ['avg', [0, 1,2,3,4,5,6,7,8,9], [10, 11, 12], [13,14,15,16,17,18,19]],
        #         [[10, 11, 12]],
        #         'timeRange': timeRange, 
        #         'minFreq': 0.05,
        #         'maxFreq': 10,
        #         'stepFreq': 0.05,  
        #         'figSize': (8,6),#(16,12), 
        #         'saveData': False, 
        #         'saveFig': filename[:-4]+'_LFP_spect_%d_%d'%(timeRange[0], timeRange[1]), 
        #         'showFig': False})
        

        # out = sim.plotting.plotLFPPSD(**{'plots': ['PSD'], 
        #         'electrodes': 
        #         #['avg', [0], [1], [2,3,4,5,6,7,8,9], [10, 11, 12], [13], [14, 15], [16,17,18,19]], 
        #         #['avg', [0, 1,2,3,4,5,6,7,8,9], [10, 11, 12], [13,14, 15,16,17,18,19]],
        #         [[10, 11, 12]],
        #         'timeRange': timeRange, 
        #         'minFreq': 0.05,
        #         'maxFreq': 10,
        #         'stepFreq': 0.05,
        #         'orderInverse': False, 
        #         'figSize': (8,6), 
        #         'rcParams': {'font.size': 20},
        #         'saveData': False, 
        #         'saveFig': filename[:-4]+'_LFP_PSD_%d_%d'%(timeRange[0], timeRange[1]), 
        #         'showFig': False})

        data = sim.analysis.preparePSD(**{
                #['avg', [0], [1], [2,3,4,5,6,7,8,9], [10, 11, 12], [13], [14, 15], [16,17,18,19]], 
                'electrodes': 
                #['avg', [0, 1,2,3,4,5,6,7,8,9], [10, 11, 12], [13,14, 15,16,17,18,19]],
                list(range(20)),
                #[[10, 11, 12]],
                'timeRange': timeRange, 
                'minFreq': 0.05,
                'maxFreq': 10,
                'stepFreq': 0.05})


        allDataLFP.append(sim.allSimData['LFP'][timeRange[0]:timeRange[1]])
        allDataPSD.append(data)

    
    #ipy.embed()

    layers = ['avg', 'infra', 'gran', 'supra']
    for i in range(20):#, layer in enumerate(layers):
        fig = plt.figure()
        for filename, d in zip(filenames, allDataPSD):
            plt.plot(d['psdFreqs'][i], d['psdSignal'][i], label=filename)
            #plt.ylim([0,0.01])
            plt.xlim([0, 1.5])
            plt.xlabel('frequency [Hz]')
            plt.ylabel('PSD [V**2/Hz]')
        plt.legend()
        plt.savefig('data/v35_batch5/psd_comparison_%s.png' % (i))
    

    # import scipy.signal as ss
    # for ichannel in range(20):
    #     fig = plt.figure()
    #     for filename, d in zip(filenames, allDataLFP):
    #         fs = 1000/sim.cfg.recordStep
    #         x = np.array(d)[:,ichannel]
    #         f, Pxx_den = ss.welch(x, fs, nperseg=4192*4)
    #         #f, Pxx_den = ss.periodogram(x, fs)
    #         #plt.semilogy(f, Pxx_den, label=filename)
    #         plt.plot(f,Pxx_den,label=filename)
    #         #plt.ylim([1e-8, 2e-5])
    #         plt.xlim([0, 1.5])
    #         plt.xlabel('frequency [Hz]')
    #         plt.ylabel('PSD [V**2/Hz]')
    #     plt.legend()
    #     plt.savefig('data/v35_batch5/psd_welch_comparison_%s.png' % (ichannel))

        # required for combined PSD plot
    #     allData.append(out[1]['allSignal'])
            
    #     with open('data/v34_batch57/v34_batch57_10sec_allData.pkl', 'wb') as f:
    #         pkl.dump({'allData': allData, 'allFreqs': out[1]['allFreqs']}, f)  
        
    # plot_LFP_PSD_combined('data/v34_batch57/v34_batch57_10sec_allData.pkl', norm=False)

    
        # # for elec in [8,9,10,11,12]:
        # #     sim.analysis.plotLFP(**{'plots': ['timeSeries'], 'electrodes': [elec], 'timeRange': [1500, 11500], 'maxFreq':50, 'figSize': (8,4), 'saveData': False, 'saveFig': filename[:-4]+'_LFP_signal_10s_elec_'+str(elec), 'showFig': False})
        # #     sim.analysis.plotLFP(**{'plots': ['PSD'], 'electrodes': [elec], 'timeRange': [1500, 11500], 'maxFreq':50, 'figSize': (8,4), 'saveData': False, 'saveFig': filename[:-4]+'_LFP_PSD_10s_elec_'+str(elec), 'showFig': False})
        # #     sim.analysis.plotLFP(**{'plots': ['spectrogram'], 'electrodes': [elec], 'timeRange': [1500, 11500], 'maxFreq':50, 'figSize': (8,4), 'saveData': False, 'saveFig': filename[:-4]+'_LFP_spec_10s_elec_'+str(elec), 'showFig': False})

        # # #plt.ion()
        # tranges = [[6000, 6200],
        #             [6000, 6300]]
        #     #         [1980, 2100],
        #     #         [2080, 2200],
        #     #         [2030, 2150],
        #     #         [2100, 2200]] 
    
        # tranges = [[x, x+200] for x in range(2400, 3500, 100)]
        # smooth = 30
        # #tranges = [[500,11500]]
        # for t in tranges:# (2100, 2200,100):    
        #     sim.analysis.plotCSD(**{
        #         'spacing_um': 100, 
        #         'layer_lines': 1, 
        #         'layer_bounds': layer_bounds, 
        #         'overlay': 'LFP',
        #         'timeRange': [t[0], t[1]], 
        #         'smooth': smooth,
        #         'saveFig': filename[:-4]+'_CSD_LFP_smooth%d_%d-%d' % (smooth, t[0], t[1]), 
        #         'vaknin': False,
        #         'figSize': (4.1,8.2), 
        #         'dpi': 300, 
        #         'showFig': 0})
        


    # ipy.embed()