# import utils
from matplotlib import pyplot as plt
from netpyne import sim ## added for lfp artifact debugging 

# allpops = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4',
# 'PV4', 'SOM4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B', 'PV5B',
# 'SOM5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'IC']

# timeRange=[1100, 1400]
# # sim = utils.loadFromFile('../data/v25_batch5/v25_batch5_0_0_2.json')
# sim.analysis.plotCSD(spacing_um=100, timeRange=timeRange, LFP_overlay=True, layer_lines=True, saveFig=1, showFig=0)
# plt.savefig('v25_batch5_0_0_2_1100-1400.png')   




## from lfp debug attempt -- now in different script (lfpDebug.py)

#sim = utils.loadFromFile('../data/lfpSimFiles/A1_v34_batch27_v34_batch27_0_1.pkl')

# fn = '../data/lfpSimFiles/A1_v34_batch27_v34_batch27_0_1.pkl'
# sim.load(fn,instantiate=False) # fn should be .pkl netpyne sim file 

fn = '../data/simDataFiles/spont/v34_batch67_CINECA/data_pklFiles/v34_batch67_CINECA_0_0_data.pkl'

sim.load(fn,instantiate=False)

layer_bounds= {'L1': 100, 'L2': 160, 'L3': 950, 'L4': 1250, 'L5A': 1334, 'L5B': 1550, 'L6': 2000}
timeRange = [1100, 1400]


CSDData, LFPData, sampr, spacing_um, dt = sim.analysis.prepareCSD(
            sim=sim,
            pop=None,
            dt=None, 
            sampr=None,
            spacing_um=100,
            norm=True,
            getAllData=True)#,
           # **kwargs)

# sim.plotting.plotCSD(spacing_um=100, layerBounds=layer_bounds, timeRange = timeRange)

# sim.plotting.plotCSD(**{
#             'spacing_um': 100, 
#             'layer_lines': 1, 
#             'layer_bounds': layer_bounds, 
#             'overlay': 'LFP',
#             'timeRange': timeRange, 
#             'smooth': 30,
#             'saveFig': fn[:-4]+'_CSD_LFP_%d-%d' % (timeRange[0], timeRange[1]), 
#             'figSize': (9, 12), 
#             'dpi': 300, 
#             'showFig': 0})
        # ax = plt.gca()
        # ax.set_xticks([1000, 1500, 2000], ['1.0', '1.5', '2.0'])#, fontsize=fontsize, fontname='Arial')

