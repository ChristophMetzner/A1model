import utils
from matplotlib import pyplot as plt

allpops = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4',
'PV4', 'SOM4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B', 'PV5B',
'SOM5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'IC']

timeRange=[1100, 1400]
sim = utils.loadFromFile('../data/v25_batch5/v25_batch5_0_0_2.json')
sim.analysis.plotCSD(spacing_um=100, timeRange=timeRange, LFP_overlay=True, layer_lines=True, saveFig=1, showFig=0)
plt.savefig('v25_batch5_0_0_2_1100-1400.png')   