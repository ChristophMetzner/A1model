from netpyne import sim 
from netpyne.analysis.tools import *
import matplotlib.pyplot as plt 
import numpy as np
# import seaborn as sns 
# import pandas as pd

basedir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/BBN/BBN_CINECA_v36_5656BF_SOA624/'#BBN_CINECA_v36_5656BF_SOA850/'
# basedir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/BBN/BBN_CINECA_v36_5656BF_singleStim/'#BBN_CINECA_v36_5656BF_SOA850/'
#basedir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/tone/pureTone_CINECA_v36_CF500_tone500_SOA624/'
#basedir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/tone/pureTone_CINECA_v36_CF500_tone500_SOA200/'

filename = 'BBN_CINECA_v36_5656BF_624ms_data.pkl' #'BBN_CINECA_v36_5656BF_850ms_data.pkl' #'pureTone_CINECA_v36_CF500_tone500_SOA624_data.pkl' 
#filename = 'pureTone_CINECA_v36_CF500_tone500_SOA200_data.pkl' 
#filename = 'REDO_BBN_CINECA_v36_5656BF_624SOA_0_0_0_data.pkl'


fn = basedir + filename


# Load simulation data 
sim.load(fn, instantiate=False)


# Get stimulus input times 
stimTimes = sim.cfg.ICThalInput['startTime']
if type(stimTimes) is not list:
	stimTimes = [stimTimes]



# ## CALCULATE IC SPIKE TIMES FOR BAR PLOT ## 
# cells_IC, cellGids_IC, netStimLabels_IC = getInclude(['IC'])	# getInclude(include)

# IC_stimTimeRanges = {}
# for stimTime in stimTimes:
# 	stimTimeKey = 'stim_' + str(int(stimTime)) + 'ms'
# 	IC_stimTimeRanges[stimTimeKey] = [stimTime, stimTime + 50]


# spkts_IC = {}
# for i in IC_stimTimeRanges.keys():
# 	timeRange = IC_stimTimeRanges[i]
# 	sel, spkts, spkgids = getSpktSpkid(cellGids_IC, timeRange=timeRange) # using [] is faster for all cells
# 	spkts_IC[i] = {'sel': sel, 'spkts': spkts, 'spkgids': spkgids, 'timeRange': timeRange}
# ### SEL IS THE PANDAS DATA FRAME WE MIGHT WANT TO USE

# numStims = len(IC_stimTimeRanges.keys())
# x = list(np.arange(1,numStims+1,1))

# IC_spkNumsList = []
# for q in spkts_IC.keys():
# 	IC_spkNumsList.append(len(spkts_IC[q]['spkts']))



# ## PLOT IC SPIKE BAR BLOT ON BOTTOM SUBPLOT ## 
# fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False)
# ax2.bar(x, IC_spkNumsList)
# #ax2.title('IC spikes')
# #plt.bar(x, IC_spkNumsList) #, label=stimTimes)			# # --> NEXT GET TITLE, LABELS ALL SORTED OUT! 
# plt.show()


##
# include=['CT6']
# cells, cellGids, netStimLabels = getInclude(include)
# sel, spkts, spkgids = getSpktSpkid(cellGids, timeRange=timeRange) # using [] is faster for all cells
# sel, spkts, spkgids = getSpktSpkid(cellGids=[] if include == ['allCells'] else cellGids, timeRange=timeRange) # using [] is faster for all cells
###########


## LISTS OF POPS ## 
EThalPops = ['TC', 'TCM']
Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'CT5B' , 'PT5B', 'IT6', 'CT6']  # all layers


##  Designate which POPS to include & the timeRange to plot  ## 
includePops = Epops 	#['IT6']	#['CT6']
timeRange = [stimTimes[0]-1250, stimTimes[-1]+50]  #[stimTimes[0]-50, stimTimes[-1]+50]


## PLOT HISTOGRAM USING HISTDATA ##
histData = sim.analysis.prepareSpikeHist(
	timeRange=timeRange, include=includePops)	# Epops		# ['CT6', 'IT6', 'CT5B', 'IT5B']

sim.plotting.plotSpikeHist(histData=histData, include='allCells',binSize=50, showFig=0, legend=True, allCellsColor='darkblue', density=False,histType='stepfilled')		# #rcParams = {'xtick.color': 'blue'}#, 'axes.facecolor': 'black'} # rcParams=rcParams # density=True	# xlabel='TIME (ms)'

plt.vlines(stimTimes, 0, 500, linestyles='dashed', colors='yellow') #(5000, 0, 500, linestyles='dashed', colors='yellow') #(stimTimes, 0, 50, linestyles='dashed', colors='yellow')  # 0, 0.000300, 

plt.show()


## OR PLOT HISTOGRAM DIRECTLY ## 
# sim.plotting.plotSpikeHist(include=includePops, timeRange = timeRange, binSize=5, showFig=0, stacked=False, legend='labels')
# plt.vlines(stimTimes, 50, 250, linestyles='dashed', colors='blue')
# plt.show()



########## ATTEMPT TO USE PD / SEABORN TO GET ENVELOPE OF HISTOGRAM ############

# binSize=50
# spkTimes = histData['spkTimes']
# histoData = np.histogram(spkTimes, bins=np.arange(timeRange[0], timeRange[1], binSize))
# histoCount = histoData[0]
# histoBins = histoData[1]#[0:-1]
# plt.figure()
# plt.stairs(histCount, histoBins)
#plt.hist(histoBins[:-1], histoBins, histtype='step', weights=histoCount)


# #plt.hist(histoCount, histoBins, histtype='step', weights=histoCount)
# df = pd.DataFrame({'x': histoBins, 'y': histoCount}) # maybe 'x' instead of 'spkTimes' ??
# sns.kdeplot(histoCount, histoBins)
# plt.show()


# df = pd.DataFrame({'spkTimes': spkTimes, 'bins': histoBins}) # maybe 'x' instead of 'spkTimes' ??
# sns.kdeplot(df)


# histoBins = pd.Series(histoData[1])
# histoCount = pd.Series(histoData[0])
# v3 = np.concatenate((histoBins,histoCount))
# sns.kdeplot(v3)
# histoData_forPD = np.concatenate(histoBins,histoCount)
# histoData_df = pd.DataFrame(histoData_forPD)
# # histoBins = histoData[1]
# # histoCount = histoData[0]
# # histoData_forPD = np.concatenate(histoBins,histoCount)
# # histoData_df = pd.DataFrame(histoData_forPD)


# sns.kdeplot(histoData_df)

# plt.show()



