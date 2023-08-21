"""
paper.py 

Paper figures

Contributors: salvadordura@gmail.com
"""

from numpy.lib.financial import pv
import utils
import json
import numpy as np
import scipy
from matplotlib import pyplot as plt
import pandas as pd
import os
import pickle
import batchAnalysis as ba
from netpyne.support.scalebar import add_scalebar
from matplotlib import cm
from bicolormap import bicolormap 
from netpyne import analysis
from matplotlib.colors import LogNorm
import IPython as ipy
from netpyne import sim

try:
    basestring
except NameError:
    basestring = str

# ---------------------------------------------------------------------------------------------------------------
# Population params
allpops = ['NGF1', 'IT2', 'SOM2', 'PV2', 'VIP2', 'NGF2', 'IT3',  'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'SOM4', 'PV4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'SOM5A', 'PV5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B',  'SOM5B', 'PV5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'SOM6', 'PV6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM', 'IC']
colorList = analysis.utils.colorList
popColor = {}
for i,pop in enumerate(allpops):
    popColor[pop] = colorList[i]


def loadSimData(dataFolder, batchLabel, simLabel):
    ''' load sim file'''
    root = dataFolder+batchLabel+'/'
    sim,data,out = None, None, None
    if isinstance(simLabel, str): 
        filename = root+simLabel+'.pkl'
        print(filename)
        sim,data,out = utils.plotsFromFile(filename, raster=0, stats=0, rates=0, syncs=0, hist=0, psd=0, traces=0, grang=0, plotAll=0)
    
    return sim, data, out, root

def axisFontSize(ax, fontsize):
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 



def plot_lfp_pop_dist():

    depthum = {0: 500, 1:1000, 2:1500} 

    for depth in [2]:
        #depth = 1
        sim.load('../data/v34_batch52/v34_batch52_0_0_%d_data.pkl' % (depth), instantiate=0)
        d = {}
        for p,v in sim.allSimData['LFPPops'].items():
            d[p] = np.max(np.abs(v[5000:15000]),0)
        df=pd.DataFrame(d)
        df=df.drop(['SOM3'],axis=1) #['NGF1', 'SOM2', 'PV2', 'VIP2', 'NGF2', 'VIP4', 'NGF4', 'VIP5A', 'NGF5A', 'VIP5B', 'NGF5B','VIP6', 'TC','TCM','HTC','IRE','IREM','TI','TIM'],axis=1)
        df=df.T
        df.columns=[0,200,400,600,800,1000,1200,1400,1600,1800,2000]
        df.columns=[x/1000.0 for x in[0,200,400,600,800,1000,1200,1400,1600,1800,2000]]

        dfuv=df*1000
        dfuvlog10=np.log10(dfuv)

        dfuvnorm = dfuv.div(dfuv.max(axis=1), axis=0)

        filtFreq=200

        #ipy.embed()

        # plot, data = sim.analysis.plotLFP(plots=['timeSeries'], filtFreq=filtFreq, figSize=(8,3), timeRange=[500,1500], electrodes=[5], saveFig='../data/v34_batch52/allpops_LFP_%d_elec_1mm.png' % (depthum[depth]))
        # scipy.io.savemat('../data/v34_batch52/LFP_hor-1000um_depth-500um.mat', {'LFP': data['LFP'], 't': data['t']})
        # ipy.embed() 

        
        #plot raster
        f, ax = plt.subplots(figsize=(11, 9))
        allpops = ['NGF1', 'IT2', 'SOM2', 'PV2', 'VIP2', 'NGF2', 'IT3',  'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'SOM4', 'PV4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'SOM5A', 'PV5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B',  'SOM5B', 'PV5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'SOM6', 'PV6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM', 'IC']
        sim.analysis.plotRaster(**{'include': allpops, 'saveFig':1, 'fileName': '../data/v34_batch51/raster.png', 'showFig': False, 'popRates': 'minimal', 'orderInverse': True, 'timeRange': [0,1000], 'figSize': (11,9), 'lw': 0.3, 'markerSize': 3, 'marker': '.', 'fontsize':18, 'dpi': 300})
    
        # # #plot lfp
        #L3
        sim.analysis.plotLFP(plots=['timeSeries'], timeRange=[500,1500], electrodes=['all'], saveFig='../data/v34_batch52/allpops_LFP_500.png')
        sim.analysis.plotLFP(plots=['timeSeries'], timeRange=[500,1500], pop='IT3', electrodes=['all'], saveFig='../data/v34_batch52/L3_pyramidalIT_LFP_500.png')
        sim.analysis.plotLFP(plots=['timeSeries'], timeRange=[500,1500], pop='PV3', electrodes=['all'], saveFig='../data/v34_batch52/L3_interneuron_LFP_500.png')

        #L3
        if depth==0:
            sim.analysis.plotLFP(plots=['timeSeries'], filtFreq=filtFreq, figSize=(8,3), timeRange=[500,1500], electrodes=[0], saveFig='../data/v34_batch52/allpops_LFP_500_elec0.png')
            sim.analysis.plotLFP(plots=['timeSeries'], filtFreq=filtFreq, figSize=(8,3), timeRange=[500,1500], pop='IT3', electrodes=[0], saveFig='../data/v34_batch52/L3_pyramidalIT_LFP_500_elec0.png')
            sim.analysis.plotLFP(plots=['timeSeries'], filtFreq=filtFreq, figSize=(8,3), timeRange=[500,1500], pop='PV3', electrodes=[0], saveFig='../data/v34_batch52/L3_interneuronPV_LFP_500_elec0.png')

        # # # L4
        if depth==1:
            sim.analysis.plotLFP(plots=['timeSeries'], filtFreq=filtFreq, figSize=(8,3), timeRange=[500,1500], electrodes=[0], saveFig='../data/v34_batch52/allpops_LFP_1000_elec0.png')
            sim.analysis.plotLFP(plots=['timeSeries'], filtFreq=filtFreq, figSize=(8,3), timeRange=[500,1500], pop='ITP4', electrodes=[0], saveFig='../data/v34_batch52/L4ITP_pyramidalIT_LFP_1000_elec0.png')
            sim.analysis.plotLFP(plots=['timeSeries'], filtFreq=filtFreq, figSize=(8,3), timeRange=[500,1500], pop='ITS4', electrodes=[0], saveFig='../data/v34_batch52/L4ITS_pyramidalCT_LFP_1000_elec0.png')

        # # L5/L6
        if depth==2:
            sim.analysis.plotLFP(plots=['timeSeries'], filtFreq=filtFreq, figSize=(8,3), timeRange=[500,1500], fontSize=18, electrodes=[0], saveFig='../data/v34_batch52/allpops_LFP_1500_elec0.png')
            sim.analysis.plotLFP(plots=['timeSeries'], filtFreq=filtFreq, figSize=(8,3), timeRange=[500,1500], fontSize=18, pop='IT5B', electrodes=[0], saveFig='../data/v34_batch52/L5B_pyramidalIT_LFP_1500_elec0.png')
            sim.analysis.plotLFP(plots=['timeSeries'], filtFreq=filtFreq, figSize=(8,3), timeRange=[500,1500], fontSize=18, pop='CT6', electrodes=[0], saveFig='../data/v34_batch52/L6_pyramidalCT_LFP_1500_elec0.png')


        # import seaborn as sns
        sns.set(font_scale=2.5)

        # plot heatmap
        f, ax = plt.subplots(figsize=(10, 10))
        p = sns.heatmap(dfuv, cmap='jet',square=True, cbar_kws={'label': 'LFP peak amplitude (uV)'},  norm=LogNorm())
        p.set_xlabel("Horizontal distance (mm)")
        p.set_xticks([0.5, 5.5, 10.5])
        p.set_xticklabels([0, 1.0, 2.0])
        p.set_ylabel("Population")
        plt.subplots_adjust(left=0.18)
        plt.savefig('../data/v34_batch52/heatmap_pop_lfp_dist_depth-%d.png' % (depthum[depth]))

        #plot heatmap norm
        f, ax = plt.subplots(figsize=(10, 10))
        p = sns.heatmap(dfuvnorm, cmap='jet', square=True, cbar_kws={'label': 'LFP normalized amplitude'},  norm=LogNorm())
        p.set_xlabel("Horizontal distance (mm)")
        p.set_xticks([0.5, 5.5, 10.5])
        p.set_xticklabels([0, 1.0, 2.0])
        p.set_ylabel("Population")
        plt.subplots_adjust(left=0.2)
        plt.savefig('../data/v34_batch52/heatmap_pop_lfp_dist_depth-%d_norm.png' % (depthum[depth]))

        


def fig_raster(batchLabel, simLabel):
    dataFolder = '../data/'
    #batchLabel = 'v34_batch49' #v34_batch27/'
    #simLabel = 'v34_batch27_0_0'

    sim, data, out, root = loadSimData(dataFolder, batchLabel, simLabel)

    timeRange = [500, 1500] #[2000, 4000]
    
    #raster
    include = allpops
    orderBy = ['pop'] #, 'y']
    #filename = '%s%s_raster_%d_%d_%s.png'%(root, simLabel, timeRange[0], timeRange[1], orderBy)
    fig1 = sim.analysis.plotRaster(include=['allCells'], timeRange=timeRange, labels='legend', 
        popRates=False, orderInverse=True, lw=0, markerSize=12, marker='.',  
        showFig=0, saveFig=0, figSize=(9*0.95, 13*0.9), orderBy=orderBy)# 
    ax = plt.gca()

    [i.set_linewidth(0.5) for i in ax.spines.values()] # make border thinner
    #plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')  #remove ticks
    #plt.xticks([1000, 2000, 3000, 4000, 5000, 6000], ['1', '2', '3', '4', '5', '6'])
    #plt.yticks([0, 5000, 10000], [0, 5000, 10000])
    
    plt.ylabel('Neuron ID') #Neurons (ordered by NCD within each pop)')
    plt.xlabel('Time (ms)')
    
    plt.title('')
    filename='%s%s_raster_%d_%d_%s_DISC_grant.png'%(root, simLabel, timeRange[0], timeRange[1], orderBy)
    plt.savefig(filename, dpi=300)




def fig_traces(batchLabel, simLabel):
    dataFolder = '../data/'
    #batchLabel = 'v34_batch49' #v34_batch27/'
    #simLabel = 'v34_batch27_0_0'

    sim, data, out, root = loadSimData(dataFolder, batchLabel, simLabel)
    #popParamLabels = list(data['simData']['popRates'])

    allpops = ['NGF1', 'IT2', 'SOM2', 'PV2', 'VIP2', 'NGF2', 'IT3',  'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'SOM4', 'PV4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'SOM5A', 'PV5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B',  'SOM5B', 'PV5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'SOM6', 'PV6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM', 'IC']

    firingpops = ['NGF1', 'PV2', 'VIP2', 'NGF2', 'IT3',  'SOM3',  'VIP3', 'NGF3', 'ITP4', 'ITS4', 'SOM4', 'PV4', 'VIP4', 'IT5A', 'CT5A', 'SOM5A', 'PV5A', 'VIP5A', 'NGF5A', 'IT5B', 'CT5B',  'PV5B', 'VIP5B', 'NGF5B',  'VIP6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI']

    firingpops = ['NGF1', 'PV2', 'NGF2', 'IT3',  'SOM3',  'VIP3', 'ITP4', 'ITS4', 'SOM4', 'PV4', 'IT5A', 'CT5A', 'SOM5A', 'VIP5A', 'IT5B', 'CT5B',  'PV5B', 'NGF5B', 'TC', 'HTC', 'IRE', 'TI']

    firingpops = ['IT3', 'CT5B', 'PV5A']


    timeRanges = [[500,1500]] #[[3000, 5000], [5000, 7000], [7000, 9000], [9000,11000]]

    cellNames = data['simData']['V_soma'].keys()
    popCells = {}
    for popName,cellName in zip(allpops,cellNames):
        popCells[popName] = cellName

    fontsiz = 20   
    for timeRange in timeRanges:

        plt.figure(figsize=(9*1.05, 1.2*2)) 
        time = np.linspace(timeRange[0], timeRange[1], 10001)
        plt.ylabel('V (mV)', fontsize=fontsiz)
        plt.xlabel('Time (ms)', fontsize=fontsiz)
        plt.xlim(500, 1500)
        # plt.ylim(-80, -30)
        plt.ylim(-120*len(firingpops),20)
        plt.yticks(np.arange(-120*len(firingpops)+60,60,120), firingpops[::-1], fontsize=fontsiz)
        #plt.xticks([0,1000,2000], [1,2,3], fontsize=fontsiz)
        #ipy.embed()

        number = 0
        
        for popName in firingpops: #allpops:
            cellName = popCells[popName]   
            Vt = np.array(data['simData']['V_soma'][cellName][timeRange[0]*10:(timeRange[1]*10)+1])
            plt.plot(time, (Vt-number*120.0), color=popColor[popName]) 
            number = number + 1

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        filename='%s%s_%d_%d_firingTraces_DISC_grant_v2.png'%(root, simLabel, timeRange[0], timeRange[1])
        plt.savefig(filename, facecolor = 'white', bbox_inches='tight' , dpi=300)



def plot_dipole():
    from netpyne import sim
    sim.load('../data/v34_batch53/v34_batch53_0_0_data.pkl')

    popDipoles = ['IT3', 'CT5B']# 'PT5B', 'CT5B', 'IT5B', 'IT6']

    sim.allSimData['dipoleSum'][1203,:] = sim.allSimData['dipoleSum'][1204,:]
    
    for i in range(len(sim.allSimData['dipoleCells'])):
        sim.allSimData['dipoleCells'][i][1203,:] = sim.allSimData['dipoleCells'][i][1204,:]

    import matplotlib; matplotlib.rcParams.update({'font.size': 16})

    for pop in popDipoles:
        maxCellGid = np.argmax([abs(sim.allSimData['dipoleCells'][i]).max() for i in sim.net.pops[pop].cellGids])
        sim.analysis.plotDipole(showCell=maxCellGid, timeRange=[500,1500], figSize=(8,3), saveFig='../data/v34_batch53/dipole_pop_%s_cell_%d_max' % (pop, maxCellGid)) # max  cell

    sim.analysis.plotDipole(saveFig='../data/v34_batch53/dipole_all', timeRange=[500,1500], figSize=(8,3))
    

    ipy.embed()



def plotEEG(sim, showCell=None, showPop=None,  timeRange=None, dipole_location='parietal_lobe', dpi=300, figSize=(19,10), showFig=True, saveFig=True):

    from lfpykit.eegmegcalc import NYHeadModel

    nyhead = NYHeadModel()

    #dipole_location = 'parietal_lobe'  # predefined location from NYHead class
    nyhead.set_dipole_pos(dipole_location)
    M = nyhead.get_transformation_matrix()

    if showCell:
        p = sim.allSimData['dipoleCells'][showCell]    
    elif showPop:
        p = sim.allSimData['dipolePops'][showPop]    
    else:
        p = sim.allSimData['dipoleSum']

    if timeRange is None:
        timeRange = [0, sim.cfg.duration]

    timeSteps = [int(timeRange[0]/sim.cfg.recordStep), int(timeRange[1]/sim.cfg.recordStep)]

    p = np.array(p).T
    
    p=p[:, timeSteps[0]:timeSteps[1]]

    # We rotate current dipole moment to be oriented along the normal vector of cortex
    p = nyhead.rotate_dipole_to_surface_normal(p)
    eeg = M @ p * 1E6 # [mV] -> [nV] unit conversion

    # plot EEG daa
    x_lim = [-100, 100]
    y_lim = [-130, 100]
    z_lim = [-160, 120]

    t = np.arange(timeRange[0], timeRange[1], sim.cfg.recordStep)

    plt.close("all")
    fig = plt.figure(figsize=[9, 16])
    fig.subplots_adjust(top=0.96, bottom=0.05, hspace=0.17, wspace=0.3, left=0.1, right=0.99)
    ax_eeg = fig.add_subplot(313, xlabel="Time (ms)", ylabel='nV', title='EEG at all electrodes')
    ax3 = fig.add_subplot(312)
    ax7 = fig.add_subplot(311, aspect=1, xlabel="x (mm)", ylabel='y (mm)',
                        xlim=y_lim, ylim=x_lim)

    dist, closest_elec_idx = nyhead.find_closest_electrode()
    print("Closest electrode to dipole: {:1.2f} mm".format(dist))

    max_elec_idx = np.argmax(np.std(eeg, axis=1))
    time_idx = np.argmax(np.abs(eeg[max_elec_idx]))
    max_eeg = np.max(np.abs(eeg[:, time_idx]))
    max_eeg_idx = np.argmax(np.abs(eeg[:, time_idx]))
    max_eeg_pos = nyhead.elecs[:3, max_eeg_idx]

    for idx in range(eeg.shape[0]):
        ax_eeg.plot(t, eeg[idx, :], c='gray') 
    ax_eeg.plot(t, eeg[closest_elec_idx, :], c='green', lw=2)

    vmax = np.max(np.abs(eeg[:, time_idx]))
    v_range = vmax
    cmap = lambda v: plt.cm.bwr((v + vmax) / (2*vmax))
    threshold = 2

    xy_plane_idxs = np.where(np.abs(nyhead.cortex[2, :] - nyhead.dipole_pos[2]) < threshold)[0]

    for idx in range(eeg.shape[0]):
        c = cmap(eeg[idx, time_idx])
        ax7.plot(nyhead.elecs[1, idx], nyhead.elecs[0, idx], 'o', ms=10, c=c, 
                zorder=nyhead.elecs[2, idx])

    img = ax3.imshow([[], []], origin="lower", vmin=-vmax,
                    vmax=vmax, cmap=plt.cm.bwr)
    cbar=plt.colorbar(img, ax=ax7, shrink=0.5)
    cbar.ax.set_ylabel('nV', rotation=90)

    ax7.plot(nyhead.dipole_pos[1], nyhead.dipole_pos[0], '*', ms=15, color='orange', zorder=1000)


    # save figure
    if saveFig:
        if isinstance(saveFig, basestring):
            filename = saveFig
        else:
            filename = sim.cfg.filename + '_EEG.png'
        try:
            plt.savefig(filename, dpi=dpi)
        except:
            plt.savefig('EEG_fig.png', dpi=dpi)


    # display figure
    if showFig is True:
        plt.show()

    import IPython as ipy; ipy.embed() 

def plot_eeg():

    from netpyne import sim
    sim.load('../data/v34_batch53/v34_batch53_0_0_data.pkl', instantiate=False)
    sim.allSimData['dipoleSum'][1203,:] = sim.allSimData['dipoleSum'][1204,:]

    import matplotlib; matplotlib.rcParams.update({'font.size': 16})

    # temp
    sim.analysis.plotDipole(saveFig='../data/v34_batch53/dipole_all', timeRange=[500,1500], figSize=(9,4.5))

    #plotEEG(sim, saveFig='../data/v34_batch53/EEG_all', timeRange=[500,1500])

def save_dipoles_matlab():
    from netpyne import sim
    sim.load('../data/v34_batch53/v34_batch53_0_0_data.pkl', instantiate=False)
    cellPos = [[c.tags['x'], sim.net.params.sizeY-c.tags['y'],c.tags['z']] for c in sim.net.cells if c.gid <=12186]
    cellDipoles = [sim.allSimData['dipoleCells'][i] for i in range(12187)]
    cellPops = [c.tags['pop'] for c in sim.net.cells if c.gid <=12186]
    matDat = {'cellPos': cellPos, 'cellPops': cellPops, 'cellDipoles': cellDipoles, 'popDipoles': sim.allSimData['dipolePops'], 'dipoleSum': sim.allSimData['dipoleSum']}
    scipy.io.savemat('A1_cell_dipoles.mat', matDat)

# Main
if __name__ == '__main__':
    #plot_dipole()
    plot_eeg()
    #fig_raster('v34_batch52', 'v34_batch52_0_0_0_data')
    #fig_traces('v34_batch27', 'v34_batch27_0_0')
    #plot_lfp_pop_dist()