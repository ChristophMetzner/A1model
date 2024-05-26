from netpyne import sim 
from matplotlib import pyplot as plt 
import os 
import numpy as np

basedir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/spont/v34_batch27/'#A1_v34_batch67_v34_batch67_0_0_data.pkl'#v34_batch27/'
# basedir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/spont/v34_batch57/'	#simDataFiles/BBN/BBN_stim_noStim/'#BBN_deltaSOA_variedStartTimes/'
# basedir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/BBN/v38_NMDAR_BBN/'
# basedir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/tone/pureTone_CINECA_v36_variedCF_variedSOA/'
# basedir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/click/click_CINECA_v36_CF4000_CF11313/'
# basedir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/click/click_CINECA_v36_CF5656_variedSOA/'
# basedir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/BBN/BBN_CINECA_variedSOA_v36/'
# basedir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/tone/pureTone_CINECA_v36/'
# basedir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/BBN/BBN_CINECA_v36_5656BF_624SOA/'
# basedir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/BBN/data_pklFiles/'
# basedir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/speech/BBN_CINECA_speech_ANmodel/'
# basedir = '../data/simDataFiles/BBN/BBN_CINECA_startTimeDebug_IPGmod/'
# '../data/simDataFiles/speech/v34_batch_eegSpeech_CINECA_trial_1/'
# fn = 'v34_batch_eegSpeech_CINECA_trial_0_2_data.pkl'
# fn = '../data/simDataFiles/speech/v34_batch_eegSpeech_CINECA_trial_0/v34_batch_eegSpeech_CINECA_trial_0_3_data.pkl'


allFiles = os.listdir(basedir)

pklFiles = []
for file in allFiles:
	if '_data.pkl' in file:
		pklFiles.append(file)


for fn in pklFiles:
	fullFilename = basedir + fn

	sim.load(fullFilename, instantiate=False)
	print('Loaded ' + fn)

	orderBy = ['pop']

	timeRange = [0,sim.cfg.duration]	#[500,sim.cfg.duration]#[500,10000]


	## FIGURING OUT GID ISSUE
	#sim.net.allPops.keys()
	# IC_cellGIDs = sim.net.allPops['IC']['cellGids']
	# print('IC_GIDs : ' + str(IC_cellGIDs))
	# IT3_cellGIDs = sim.net.allPops['IT3']['cellGids']
	# print('IT3 cell GIDs : ' + str(IT3_cellGIDs))

	# IC_spkTimes = sim.net.allPops['IC']['spkTimes']
	# print('IC spk times: ' + str(IC_spkTimes))
	# IT3_spkTimes = sim.net.allPops['IT3']['spkTimes']
	# print('IT3_spkTimes: ' + str(IT3_spkTimes))


	fig1 = sim.analysis.plotRaster(include=['allCells'], timeRange=timeRange, labels='legend', 
		popRates=False, orderInverse=True, lw=0, markerSize=12, marker='.',  
		showFig=0, saveFig=0, figSize=(9*0.95, 13*0.9))#, orderBy=orderBy)


	# # fig1 = sim.plotting.plotRaster(include=['allCells'], timeRange=timeRange, 
	# # 	popRates=False, orderInverse=True, lw=0, #markerSize=10, marker='.',  
	# # 	showFig=0, saveFig=0, figSize=(9*0.95, 13*0.9), orderBy=orderBy)

	# print('RASTER PLOTTED')

	# # ax = plt.gca()

	# # [i.set_linewidth(0.5) for i in ax.spines.values()] # make border thinner

	# ## set xticks -- but NOTE: this only works for single start time for now!! ## 
	# # xTickInterval = sim.cfg.ICThalInput['startTime']
	# # xTickMarkers = np.arange(timeRange[0], sim.cfg.duration, 2000)#500)	# xTickInterval)
	# # plt.xticks(xTickMarkers)	# plt.xticks(timeRange, [timeRange[0], timeRange[1]]) #['0', '1'])
	
	# ## set y ticks ##
	# #plt.yticks([0, 5000, 10000], [0, 5000, 10000])  	## for full-scale sim
	# #plt.yticks([0, 2500, 5000], [0, 2500, 5000])		## for half-scale sim 

	# plt.ylabel('Neuron ID') #Neurons (ordered by NCD within each pop)')
	# plt.xlabel('Time (ms)')

	# plt.title('')

	rasterFilename = fn.split('_data.pkl')[0]
	rasterFile = basedir + rasterFilename + '_RASTER.png' #'speechRaster_0_2.png'
	plt.savefig(rasterFile, dpi=300)

	print('RASTER SAVED')
