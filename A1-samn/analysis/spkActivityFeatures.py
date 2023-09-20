from netpyne import specs, sim
from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np
from matplotlib.collections import LineCollection


###################
## LOAD SIM FILE ## 
###################
spontBaseDir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/spont/v34_batch67_CINECA/data_pklFiles/'
BBNBaseDir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/BBN/'
clickBaseDir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/click/'
toneBaseDir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/tone/'

tTypeBaseDir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/spont/v37_tTypeReduction0_CINECA/'

##########################
## LOAD POP COLORS FILE ## 
##########################
with open('popColors.pkl', 'rb') as fileObj: popColors = pkl.load(fileObj)['popColors']



#####################
## POPS TO INCLUDE ##
#####################
thalPops = ['TC', 'TCM', 'HTC', 'TI', 'IRE', 'IREM', 'TIM'] 
thalMatrix = ['TCM', 'TIM', 'IREM']
thalCore = ['TC', 'HTC', 'TI', 'IRE']
thalEpops = ['TC', 'HTC', 'TCM']
thalRetic = ['IRE', 'IREM']
MGBinhib = ['TI', 'TIM']

Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'CT5B' , 'PT5B', 'IT6', 'CT6']  # all layers
Ipops = ['NGF1',                            # L1
	'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
	'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
	'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
	'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A  
	'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
	'PV6', 'SOM6', 'VIP6', 'NGF6']      # L6 

L1 = ['NGF1']
L2 = ['IT2', 'SOM2', 'PV2', 'VIP2', 'NGF2']
L3 = ['IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3']
L4 = ['ITS4', 'ITP4', 'SOM4', 'PV4', 'VIP4', 'NGF4']
L5A = ['IT5A', 'CT5A', 'SOM5A', 'PV5A', 'VIP5A', 'NGF5A']
L5B = ['IT5B', 'CT5B', 'PT5B', 'SOM5B', 'PV5B', 'VIP5B', 'NGF5B']
L6 = ['IT6', 'CT6', 'SOM6', 'PV6', 'VIP6', 'NGF6']


supraPops = L1 + L2 + L3
granPops = L4
infraPops = L5A + L5B + L6


######## CHOOSE PARAMS #############
### tone:
#toneBaseDir + 'pureTone_CINECA_v36_CF500_tone500_SOA200/pureTone_CINECA_v36_CF500_tone500_SOA200_data.pkl'
#toneBaseDir + 'pureTone_CINECA_v36_CF500_tone500_SOA624/pureTone_CINECA_v36_CF500_tone500_SOA624_data.pkl'
#toneBaseDir + 'pureTone_CINECA_v36_CF5656_tone5656_SOA200/pureTone_CINECA_v36_CF5656_tone5656_SOA200_data.pkl'
#filename = toneBaseDir + 'pureTone_CINECA_v36_CF5656_tone5656_SOA624/pureTone_CINECA_v36_CF5656_tone5656_SOA624_data.pkl'

#filename = clickBaseDir + 'click_CINECA_v36_CF4000/click_CINECA_v36_CF4000_data.pkl'
#filename = clickBaseDir + 'click_CINECA_v36_CF5656/click_CINECA_v36_CF5656_data.pkl'
#filename = clickBaseDir + 'click_CINECA_v36_CF11313/click_CINECA_v36_CF11313_data.pkl'

# filename = spontBaseDir + 'v34_batch67_CINECA_0_0_data.pkl'

# includePopList = [['all']]#, Epops]#, thalPops] #Epops #thalPops #['all']  
# timeRange = [1000,12000]	#11500]
# filtFreq = None 



# #####################
# ## Load sim file 
# sim.load(filename, instantiate=False)


# if 'BBN' in filename:
# 	stimTimes = sim.cfg.ICThalInput['startTime']
# else:
# 	stimTimes = []



plotSingleSim = 0
plotDiffSims = 1



###############################################################
### PLOTTING PSD OF SPIKING	### 

if plotSingleSim:

	filename = spontBaseDir + 'v34_batch67_CINECA_0_0_data.pkl'

	includePopList = [['all']]
	timeRange = [1000,12000]	#11500]
	maxFreq = 80
	binSize = 1
	filtFreq = None 
	transformMethod = 'morlet'
	saveFig=0

	sim.load(filename, instantiate=False)

	for includePops in includePopList:
		## PLOT PSD OF SPIKING
		fig, psdData = sim.analysis.spikes_legacy.plotRatePSD(include=includePops,timeRange=timeRange, transformMethod=transformMethod, showFig=0, maxFreq=maxFreq, binSize=binSize, popColors=popColors)
		PSDtitle = 'PSD of spike rate activity for ' + filename.split('/')[-1][:-4]
		plt.title(PSDtitle)

		if saveFig:
			## SAVING
			saveBaseDir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/spkRatePSDs/'
			popsForSave = ''
			if includePops == Epops:
				popStr = '_Epops'
			elif includePops == thalPops:
				popStr = '_thalPops'
			elif includePops == ['all']:
				popStr = '_overallNetwork'
			elif includePops == ['eachPop', 'allCells']:
				popStr = '_eachPop'
			else:
				popStr = ''
				for pop in includePops:
					popStrPart = '_' + pop
					popStr += popStrPart

			print(popStr)


			saveFileName = filename.split('/')[-1][:-4] + popStr
			saveDir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/spkRatePSDs/'
			savePath = saveDir + saveFileName
			plt.savefig(savePath)

		else:
			## SHOW FIGS 
			plt.show()






###############################################################
### WORKING WITH ALL FREQS / ALL SIGNAL DATA ### 

# freqBandRanges = {'delta': [1,4], 'theta': [4,8], 'alpha': [8,13], 'beta': [13,30], 'gamma': [30,80], 'hgamma': [80,300]}
freqBandRanges = {'delta': [0.1,4.1], 'theta': [4.1,8.1], 'alpha': [8,13], 'beta': [13,30], 'gamma': [30,80], 'hgamma': [80,300]}
bandColors = {'delta': 'red', 'theta': 'green', 'alpha': 'blue', 'beta': 'yellow', 'gamma': 'orange', 'hgamma': 'pink'}
x_tick_locations = {'delta': 2.05, 'theta': 6, 'alpha': 10.5, 'beta': 21.5, 'gamma': 55, 'hgamma': 190} 

if plotDiffSims:

	baseFile = spontBaseDir + 'v34_batch67_CINECA_0_0_data.pkl'    #BBNBaseDir + 'BBN_deltaSOA_variedStartTimes/BBN_deltaSOA_variedStartTimes_2_data.pkl' #'v39_BBN_E_to_I/v39_BBN_E_to_I_1_0_data.pkl' #'BBN_deltaSOA_variedStartTimes/BBN_deltaSOA_variedStartTimes_2_data.pkl' #spontBaseDir + 'v34_batch67_CINECA_0_0_data.pkl'  #spontBaseDir + 'v34_batch67_CINECA_0_0_data.pkl' # BBNBaseDir + 'v38_NMDAR_BBN/v38_NMDAR_BBN_0_0_data.pkl'  #BBNBaseDir + 'v38_NMDAR_BBN/v38_NMDAR_BBN_0_0_data.pkl'  #	# 
	compareFile = BBNBaseDir + 'BBN_deltaSOA_variedStartTimes/BBN_deltaSOA_variedStartTimes_0_data.pkl'	# 'v38_NMDAR_BBN/v38_NMDAR_BBN_0_0_data.pkl' # BBNBaseDir + 'v38_NMDAR_BBN/v38_NMDAR_BBN_0_1_data.pkl' #'BBN_synWeightFractionEI_customCort/BBN_synWeightFractionEI_customCort_0_1_data.pkl'  #'v39_BBN_E_to_I/v39_BBN_E_to_I_1_2_data.pkl'  #'v38_NMDAR_BBN/v38_NMDAR_BBN_0_1_data.pkl' #BBNBaseDir + 'BBN_deltaSOA_variedStartTimes/BBN_deltaSOA_variedStartTimes_2_data.pkl' #BBNBaseDir + 'BBN_deltaSOA_variedStartTimes/BBN_deltaSOA_variedStartTimes_0_data.pkl'	 #BBNBaseDir + 'BBN_CINECA_v36_5656BF_SOA624/BBN_CINECA_v36_5656BF_624ms_data.pkl'		# BBNBaseDir + 'v38_NMDAR_BBN/v38_NMDAR_BBN_0_0_data.pkl'	# BBNBaseDir + 'BBN_CINECA_v36_5656BF_SOA624/BBN_CINECA_v36_5656BF_624ms_data.pkl'

	comparisonFiles = [baseFile, compareFile]


	includePopList = [['all']]
	#timeRange = [0, sim.cfg.duration]# [1000,11500]	#11500]
	minFreq = freqBandRanges['delta'][0]  # 0.8 #
	maxFreq = freqBandRanges['beta'][1] #freqBandRanges['theta'][1]
	binSize = 0.1 #0.5  #0.2 # 1
	filtFreq = None 
	norm=False#True#
	transformMethod = 'morlet'



	comparisonFilesData = {}

	baseKey = baseFile.split('/')[-1][:-4]
	compareKey = compareFile.split('/')[-1][:-4]

	comparisonFilesData[baseKey] = {}
	comparisonFilesData[compareKey] = {}

	# Generate legend labels 
	fileTypes = ['spont', 'BBN', 'click', 'tone']
	baseLabelList = [x for x in fileTypes if x in baseFile]
	baseLabel = baseLabelList[0]
	compareLabelList = [y for y in fileTypes if y in compareFile]
	compareLabel = compareLabelList[0]
	## MANUALLY SET BASE AND COMPARE LABELS
	baseLabel = 'spont WT' # 'BBN WT'# # 'spont WT' #
	compareLabel = 'BBN WT, 300ms ISI (delta)' # 'BBN NMDAR, 300ms ISI (delta)'  ##	# #624ms ISI'#300ms ISI (delta)'


	### GET BASE FILE DATA ## 
	sim.load(baseFile, instantiate=False)
	timeRange = [0, sim.cfg.duration]	##[100, sim.cfg.duration]#[1000, 11500]#sim.cfg.duration]
	baseFig, basePSDdata = sim.analysis.spikes_legacy.plotRatePSD(include=['all'],timeRange=timeRange, showFig=0, minFreq=minFreq, maxFreq=maxFreq, binSize=binSize, stepFreq=binSize, popColors=popColors, norm=norm)
	allFreqs = basePSDdata['allFreqs'][0]  			# [1, 2, 3, 4, .... maxFreq] 		# ARRAY 
	allSignal = basePSDdata['allSignal'][0]	
	comparisonFilesData[baseKey]['allFreqs'] = allFreqs
	comparisonFilesData[baseKey]['allSignal'] = allSignal
	plt.close(baseFig)

	## GET COMPARE FILE DATA ## 
	sim.load(compareFile, instantiate=False)
	timeRange = [0, sim.cfg.duration] #[100, sim.cfg.duration]#[1000, 11500]#sim.cfg.duration]
	compareFig, comparePSDdata = sim.analysis.spikes_legacy.plotRatePSD(include=['all'],timeRange=timeRange, showFig=0, minFreq=minFreq, maxFreq=maxFreq, binSize=binSize, stepFreq=binSize, popColors=popColors, norm=norm)
	allFreqs = comparePSDdata['allFreqs'][0]  			# [1, 2, 3, 4, .... maxFreq] 		# ARRAY 
	allSignal = comparePSDdata['allSignal'][0]	
	comparisonFilesData[compareKey]['allFreqs'] = allFreqs
	comparisonFilesData[compareKey]['allSignal'] = allSignal
	plt.close(compareFig)


	## GET DIFFERENCE DATA BTWN BASE & COMPARE FILE DATA ## 
	differenceData = {}
	differenceData['allFreqs'] = allFreqs
	differenceData['allSignal'] = comparisonFilesData[compareKey]['allSignal'] - comparisonFilesData[baseKey]['allSignal']



	####################
	##### PLOTTING #####
	####################

	plotSubPlots = 0   # plot diff as subplots or overlay 
	# plotOverlayDiff = 1


	if plotSubPlots:
		print('plotting as subplots')
		fig, axs = plt.subplots(2, figsize=(10,8))

		axs[0].plot(comparisonFilesData[baseKey]['allFreqs'], comparisonFilesData[baseKey]['allSignal'], label=baseLabel, color='black')
		axs[0].plot(comparisonFilesData[compareKey]['allFreqs'], comparisonFilesData[compareKey]['allSignal'], label=compareLabel, color='gray', linestyle='--')

		## Set titles ## 
		# fig.suptitle('Spike Rate PSD for ' + baseLabel + ' & ' + compareLabel)
		axs[0].set_title('Spike Rate PSD for ' + baseLabel + ' & ' + compareLabel, fontsize=16)		#'Overlay', fontsize=16)
		axs[1].set_title('Difference', fontsize=16)

		## Set axes labels ## 
		axs[1].set_xlabel('Frequency (Hz)', fontsize=16)
		axs[0].set_ylabel('Power', fontsize=16)
		axs[1].set_ylabel(r'$\Delta$ Power', fontsize=16)

		## Add legend ## 
		axs[0].legend()

		## Add frequency band demarcations onto x-axis ## 
		colors = []
		xBands = [minFreq]
		x_ticks = []
		x_labels = []
		for band in freqBandRanges.keys():
			if freqBandRanges[band][1] <= maxFreq and freqBandRanges[band][0] >= minFreq:
				xBands.append(freqBandRanges[band][1])
				colors.append(bandColors[band])
				x_ticks.append(x_tick_locations[band])
				x_labels.append(band)
		yBands = [0] * len(xBands)  # min(differenceData['allSignal']) * len(xBands)
		points = np.array([xBands, yBands]).T.reshape(-1, 1, 2)
		segments = np.concatenate([points[:-1], points[1:]], axis=1)
		lc = LineCollection(segments, colors=colors, linewidth=2, transform=axs[1].get_xaxis_transform(), clip_on=False)
		axs[1].add_collection(lc)

		axs[1].spines["bottom"].set_visible(False)
		axs[1].spines["right"].set_visible(False)
		axs[1].spines["top"].set_visible(False)

		## Set frequency band labels ## 
		axs[1].set_xticks(ticks=x_ticks)
		axs[1].set_xticklabels(labels=x_labels)
		axs[1].tick_params(bottom=False, labelsize=10)

		## PLOT DIFFERENCE DATA ## 
		# axs[1].plot(differenceData['allFreqs'], differenceData['allSignal'], color='black', label='Diff')
		x_diff = list(differenceData['allFreqs'])
		y_diff = list(differenceData['allSignal'])

		numSegments = len(xBands) - 1

		for i in range(numSegments):
			idx0 = x_diff.index(xBands[i])
			idx1 = x_diff.index(xBands[i+1])
			segment_x = x_diff[idx0:idx1+1]
			segment_y = y_diff[idx0:idx1+1]
			axs[1].plot(segment_x, segment_y, color=colors[i])


		fig.subplots_adjust(hspace=0.5)


	else: #if plotOverlayDiff: 
		print('plotting as overlay')
		fig, axs = plt.subplots(1, figsize=(13,5))

		## Set title ## 
		fig.suptitle('Spike Rate PSD for ' + baseLabel + ' & ' + compareLabel)

		## Set axes labels ## 
		axs.set_xlabel('Frequency (Hz)', fontsize=16)
		axs.set_ylabel('Power', fontsize=16)

		### UNCOMMENT HERE TO AXS.TICK_PARAMS() IF WANT TO HAVE X-AXIS WITH FREQ BAND DEMARCATIONS ### 
		## Add frequency band demarcations onto x-axis ## 
		# colors = []
		# xBands = [minFreq]
		# x_ticks = []
		# x_labels = []
		# for band in freqBandRanges.keys():
		# 	if freqBandRanges[band][1] <= maxFreq and freqBandRanges[band][0] >= minFreq:
		# 		xBands.append(freqBandRanges[band][1])
		# 		colors.append(bandColors[band])
		# 		x_ticks.append(x_tick_locations[band])
		# 		x_labels.append(band)
		# yBands = [0] * len(xBands)  # min(differenceData['allSignal']) * len(xBands)  <-- ?? 
		# points = np.array([xBands, yBands]).T.reshape(-1, 1, 2)
		# segments = np.concatenate([points[:-1], points[1:]], axis=1)
		# lc = LineCollection(segments, colors=colors, linewidth=2, transform=axs.get_xaxis_transform(), clip_on=False)
		# axs.add_collection(lc)
		# axs.spines["bottom"].set_visible(False)

		# axs.spines["right"].set_visible(False)
		# axs.spines["top"].set_visible(False)

		## Set frequency band labels ## 
		# axs.set_xticks(ticks=x_ticks)
		# axs.set_xticklabels(labels=x_labels) 		# axs.set_xticks(xBands)
		# axs.tick_params(bottom=False, labelsize=10)

		### PLOT THE ACTUAL DATA ## 
		axs.plot(comparisonFilesData[baseKey]['allFreqs'], comparisonFilesData[baseKey]['allSignal'], label=baseLabel, color='black')
		axs.plot(comparisonFilesData[compareKey]['allFreqs'], comparisonFilesData[compareKey]['allSignal'], label=compareLabel, color='gray', linestyle='--')
		# [:10]

		## PLOT THE DIFFERENCE DATA ## 
		#### ORIG: 
		axs.plot(differenceData['allFreqs'], differenceData['allSignal'], color='red', label='Diff')  # color='green'
		start, end = axs.get_xlim()
		axs.xaxis.set_ticks(np.arange(start, end, binSize))
		#axs.xaxis.set_ticks(np.arange(0.9, 4.5, 0.2))#(np.arange(0.1, 4.5, 0.2))#np.arange(start, end, 0.3))
		# #### NEW:  ## UNCOMMENT FROM HERE TO AXS.PLOT(SEGMENT_X, ...) IF WISH TO PLOT BY FREQUENCY BAND COLOR 
		# x_diff = list(differenceData['allFreqs'])
		# y_diff = list(differenceData['allSignal'])

		# numSegments = len(xBands) - 1

		# for i in range(numSegments):
		# 	idx0 = x_diff.index(xBands[i])
		# 	idx1 = x_diff.index(xBands[i+1])
		# 	segment_x = x_diff[idx0:idx1+1]
		# 	segment_y = y_diff[idx0:idx1+1]
		# 	axs.plot(segment_x, segment_y, color=colors[i])

		## ADD LINE FOR STIM ## 
		#stimPosition=1000/150
		#axs.axvline(stimPosition) #, ymin=0, ymax=5)

		## Add legend ## 
		axs.legend()

	# SHOW PLOTS 
	plt.show()




	################


	# figNum=1
	# for file in comparisonFiles:

	# 	fileKey = file.split('/')[-1][:-4]
	# 	comparisonFilesData[fileKey] = {}

	# 	sim.load(file, instantiate=False)

	# 	# for includePops in includePopList:  ## ONLY DO THIS FOR ['all'] FOR NOW!!! figure out alternatives later.  		# include=includePops

	# 	# plt.figure(figNum)
	# 	fig, psdData = sim.analysis.spikes_legacy.plotRatePSD(include=['all'],timeRange=timeRange, showFig=0, maxFreq=80, binSize=1, popColors=popColors)

	# 	allFreqs = psdData['allFreqs'][0]  			# [1, 2, 3, 4, .... maxFreq] 		# ARRAY 
	# 	allSignal = psdData['allSignal'][0]			# ARRAY of same size as allFreqs 

	# 	comparisonFilesData[fileKey]['allFreqs'] = allFreqs
	# 	comparisonFilesData[fileKey]['allSignal'] = allSignal

	# 	fig.suptitle(fileKey)
	# 	plt.show()
	# 	figNum+=1

		# for freqBand in freqBandRanges.keys():
		# 	comparisonFilesData[fileKey][freqBand] = []

		# 	for freq, signal in zip(allFreqs, allSignal):
		# 		if freq >= freqBandRanges[freqBand][0] and freq < freqBandRanges[freqBand][1]:
		# 			comparisonFilesData[fileKey][freqBand].append([freq, signal])



	## OVERLAY PLOTS ## 
	# baseKey = baseFile.split('/')[-1][:-4]
	# compareKey = compareFile.split('/')[-1][:-4]

	# # plt.clf()
	# plt.figure(3)
	# plt.plot(comparisonFilesData[baseKey]['allFreqs'], comparisonFilesData[baseKey]['allSignal'])
	# plt.plot(comparisonFilesData[compareKey]['allFreqs'], comparisonFilesData[compareKey]['allSignal'])
	# # plt.show()




###############################################################


# ## PLOT SPIKE FREQUENCY  (SUM)
# freqData = sim.analysis.prepareSpikeHist(
# 	timeRange=timeRange, include=includePops)
# sim.plotting.plotSpikeFreq(freqData=freqData, include='allCells', binSize=1, showFig=0)#, legend=True, allCellsColor='darkblue', density=False,histType='stepfilled')		# #rcParams = {'xtick.color': 'blue'}#, 'axes.facecolor': 'black'} # rcParams=rcParams # density=True	# xlabel='TIME (ms)'
# if len(stimTimes) > 0:
# 	plt.vlines([time for time in stimTimes if time < 10000], ymin=0, ymax=100, colors='blue')


# ## PLOT SPIKE FREQUENCY  (NOT SUM)
# sim.analysis.spikes_legacy.plotSpikeHist(measure='rate', filtFreq=filtFreq,  include=includePops, timeRange=timeRange, showFig=0, figSize=(12,8), popColors=popColors) #, overlay=False)
# plt.title(str(includePops))
# if len(stimTimes) > 0:
# 	plt.vlines([time for time in stimTimes if time < 10000], ymin=0, ymax=100, colors='blue')













##### THESE DO NOT WORK 
# sim.plotting.plotSpikeFreq(include=['IT2', 'CT6'], timeRange=[1000,5000], showFig=1)#,legend=False,figSize=(10,8))
# sim.analysis.plotSpikeHist(measure='rate', timeRange=[1000,5000], showFig=1, include=['IT2', 'CT6', 'TC'])

