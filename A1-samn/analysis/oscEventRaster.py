from netpyne import sim 
from matplotlib import pyplot as plt 
import os 
import numpy as np


basedir = '../data/oscEventRasters/pklFiles/'
savedir = '../data/oscEventRasters/rasters/'
# fn = '../data/oscEventRasters/pklFiles/A1_v34_batch65_v34_batch65_0_0_data.pkl'


# allFiles = os.listdir(basedir)

# pklFiles = []
# for file in allFiles:
# 	if '_data.pkl' in file:
# 		pklFiles.append(file)


waveletInfo = {
	'Delta': {'dataFile': 'A1_v34_batch65_v34_batch65_0_0_data.pkl', 'timeRange': [1484.9, 2512.2], 
				'timeBuffer': 308.1, 'dataFile2': 'v34_batch57_3_4_data.pkl', 
				'xtick_locs': [1326.1, 1526.1, 1726.1, 1926.1, 2126.1, 2326.1, 2526.1, 2726.1], 'xtick_labels': [-600, -400, -200, 0, 200, 400, 600, 800], 
				'layer_loc': 925.56999}, 	# 'timeRange': [1484.9123743729062, 2512.2209353489225]

	'Beta': {'dataFile': 'A1_v34_batch67_v34_batch67_0_0_data.pkl',	'timeRange': [455.8, 571.4], 
				'timeBuffer': 34.6, 'dataFile2': 'v34_batch57_3_2_data.pkl',
				'xtick_locs': [430.3, 455.3, 480.3, 505.3, 530.3, 555.3, 580.3, 605.3], 'xtick_labels': [-75, -50, -25, 0, 25, 50, 75, 100], 
				'layer_loc': 393}, 	#	'timeRange': [455.80379839663993, 571.4047617460291]

	'Alpha': {'dataFile': 'A1_v34_batch67_v34_batch67_0_0_data.pkl', 'timeRange': [3111, 3324.7], 
				'timeBuffer': 64.1, 'dataFile2': 'v34_batch57_3_2_data.pkl',
				'xtick_locs': [3056.25, 3106.25, 3156.25, 3206.25, 3256.25, 3306.25, 3356.25], 'xtick_labels': [-150, -100, -50, 0, 50, 100, 150],
				'layer_loc': 2995},  # 'timeRange': [3111.0311106222125, 3324.733247664953] # NOTE: 6_11 timeRange? 

	'Theta': {'dataFile': 'A1_v34_batch67_v34_batch67_1_1_data.pkl', 'timeRange': [2785, 3347.9], 
				'timeBuffer': 168.8, 'dataFile2': 'v34_batch57_3_3_data.pkl',
				'xtick_locs': [2686.95, 2886.95, 3086.95, 3286.95, 3486.95], 'xtick_labels': [-400, -200, 0, 200, 400], 
				'layer_loc': 2477}, #2467.50999	# 'timeRange': [2785.22321038684, 3347.9278996316607]

	'Gamma': {'dataFile': 'v34_batch57_4_4_data.pkl', 'timeRange': [3895.6, 3957.2], 
				'timeBuffer': 18.4, 'dataFile2': 'v34_batch57_4_4_data.pkl',
				'xtick_locs': [3878.3, 3898.3, 3918.3, 3938.3, 3958.3], 'xtick_labels':[-40, -20, 0, 20, 40], 
				'layer_loc': 3862}}	 # 3854.38 # gamma timeRange: [3895.632463874398, 3957.13297638294]

freqBands = list(waveletInfo.keys()) # ['Theta']

replacementTest = 1
batch57 = 0
saveFigs = 0
markerType = 'o'
layerLines = 'layers' #'regions'   # OPTIONS: False, 'regions', 'layers'


replacementTestFiles = {'v34_batch67_CINECA_0_0_data.pkl': 
{'v34_batch67_CINECA_0_0_data_SIM_wavelet_chan_0_gamma_106': {'timeRange': [7687.28, 7728.23]}, 
'v34_batch67_CINECA_0_0_data_SIM_wavelet_chan_0_gamma_107': {'timeRange': [8669.99334996675, 8731.393656968285]}, 
'v34_batch67_CINECA_0_0_data_SIM_wavelet_chan_2_gamma_490': {'timeRange': [1539.1576957884788, 1605.5080275401376]},
'v34_batch67_CINECA_0_0_data_SIM_wavelet_chan_4_gamma_1001': {'timeRange': [2372.8618643093214, 2434.7121735608675]},
'v34_batch67_CINECA_0_0_data_SIM_wavelet_chan_6_alpha_1350': {'timeRange': [98.1004905024525, 413.95206976034876]},
'v34_batch67_CINECA_0_0_data_SIM_wavelet_chan_7_gamma_1699': {'timeRange': [10274.159138942596, 10328.860962397493]},
'v34_batch67_CINECA_0_0_data_SIM_wavelet_chan_8_theta_1803': {'timeRange': [7156.785783928919, 7593.437967189835]},
'v34_batch67_CINECA_0_0_data_SIM_wavelet_chan_11_theta_2461': {'timeRange': [1404.6570232851163, 1948.959744798724]},
'v34_batch67_CINECA_0_0_data_SIM_wavelet_chan_13_beta_2983': {'timeRange': [10000.0, 10237.807927195146]},
'v34_batch67_CINECA_0_0_data_SIM_wavelet_chan_13_theta_2942': {'timeRange': [1428.5071425357125, 1992.2599612998063]},
'v34_batch67_CINECA_0_0_data_SIM_wavelet_chan_19_beta_4336': {'timeRange': [2149.6607483037415, 2332.7116635583175]}}} #,
#'v34_batch67_CINECA_4_0_data.pkl': {},
#'v34_batch67_CINECA_4_4_data.pkl': {}}

replacementBaseDir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/spont/v34_batch67_CINECA/data_pklFiles/'
saveDir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/oscEventRasters/rasters/oscReplacementRasters/'
saveReplacements = 1

if replacementTest:
	for seedFile in replacementTestFiles.keys():
		fn = replacementBaseDir + seedFile
		sim.load(fn, instantiate=False)

		for png in replacementTestFiles[seedFile].keys():
			timeRange = replacementTestFiles[seedFile][png]['timeRange']

			orderBy = ['pop']

			## PLOT RASTER ## 
			sim.analysis.plotRaster(include=['allCells'], timeRange=timeRange, labels=False,
				popRates=False, orderInverse=True, lw=0, markerSize=12, marker=markerType,
				showFig=0, saveFig=0, orderBy=orderBy)#, figSize=(8,6))			#figSize=(6.4, 4.8) # labels='legend'

			plt.title(png)

			if saveReplacements:
				rasterFile = saveDir + png + '_raster.png'
				plt.savefig(rasterFile, dpi=300)
			else:
				plt.show()


if batch57: 
	for band in freqBands:
		fn = basedir + waveletInfo[band]['dataFile2']
		sim.load(fn, instantiate=False)
		print('Loaded ' + fn)

		orderBy = ['pop']

		timeRange = waveletInfo[band]['timeRange']
		timeBuffer = waveletInfo[band]['timeBuffer']
		timeRangeBuffered = [0,0]
		timeRangeBuffered[0] = timeRange[0] - timeBuffer
		timeRangeBuffered[1] = timeRange[1] + timeBuffer
		print('new timeRange: ' + str(timeRangeBuffered))


		## PLOT RASTER ## 
		sim.analysis.plotRaster(include=['allCells'], timeRange=timeRangeBuffered, labels=False,
			popRates=False, orderInverse=True, lw=0, markerSize=12, marker=markerType,
			showFig=0, saveFig=0, orderBy=orderBy, figSize=(8,6))			#figSize=(6.4, 4.8) # labels='legend'

		## Set x ticks and x labels 
		plt.xlabel('Time (ms)', fontsize=12)
		x_tick_locs = waveletInfo[band]['xtick_locs']
		x_tick_labels = waveletInfo[band]['xtick_labels']
		plt.xticks(x_tick_locs, x_tick_labels) # add in fontsize 

		## Set title of plot 
		plt.title(band + ', Oscillation Event Raster', fontsize=14)


		# vertical demarcation lines to indicate beginning and end of oscillation event #
		rasterData = sim.analysis.prepareRaster(timeRange=timeRangeBuffered, maxSpikes=1e8, orderBy=['pop'],popRates=True)
		spkInds = rasterData['spkInds']
		# print('min spkInds: ' + str(min(spkInds)))
		# print('max spkInds: ' + str(max(spkInds)))

		plt.vlines(timeRange[0], min(spkInds), max(spkInds), colors='blue', linestyles='dashed')
		plt.vlines(timeRange[1], min(spkInds), max(spkInds), colors='blue', linestyles='dashed')


		### LAYER BOUNDARIES #### 
		popLabels = rasterData['popLabels']
		popNumCells = rasterData['popNumCells']

		indPop = []
		for popLabel, popNumCell in zip(popLabels, popNumCells):
			indPop.extend(int(popNumCell) * [popLabel])


		if layerLines == 'regions':
			L1 = [0,100]			#	'NGF1'
			L4 = [0,0]				#	'ITP4' --> 'NGF4'
			L5A = [0,0]				#	'IT5A' --> 'NGF5A'
			Thal = [12000, 12907]	#	'TC' --> 'TIM'

			listL1 = []
			listL4_TOP = []
			listL5A_TOP = []
			listThal_TOP = []
			listThal_BOTTOM = []

			for spk in spkInds:
				if indPop[spk] == 'NGF1':
					listL1.append(spk)

				elif indPop[spk] == 'ITP4':
					listL4_TOP.append(spk)

				elif indPop[spk] == 'IT5A':
					listL5A_TOP.append(spk)

				elif indPop[spk] == 'TC':
					listThal_TOP.append(spk)

				elif indPop[spk] == 'TIM':
					listThal_BOTTOM.append(spk)


			L1[0] = min(listL1)
			plt.axhline(0, color='black', linestyle=":")

			L4[0] = min(listL4_TOP)
			plt.axhline(L4[0], color='black', linestyle=":")

			L5A[0] = min(listL5A_TOP)
			plt.axhline(L5A[0], color='black', linestyle=":")

			Thal[0] = min(listThal_TOP)
			Thal[1] = max(listThal_BOTTOM)
			plt.axhline(Thal[0], color='black', linestyle=":")

			## Annotate y-axis with layer region labels ## 
			text_loc = waveletInfo[band]['layer_loc']
			print('layer location, ' + band + ': ' + str(text_loc))

			supragranular_loc = (L1[0] + L4[0])/2
			plt.text(text_loc, supragranular_loc, 'SUPRA')  # x_tick_locs[0]-(timeBuffer*1.3)

			granular_loc = (L4[0] + L5A[0])/2
			plt.text(text_loc, granular_loc, 'GRAN')  # x_tick_locs[0]-(timeBuffer*1.3)

			infragranular_loc = (L5A[0] + Thal[0])/2
			plt.text(text_loc, infragranular_loc, 'INFRA')	# x_tick_locs[0]-(timeBuffer*1.3)

			thal_loc = ((Thal[0] + Thal[1]) / 2)*1.025
			plt.text(text_loc, thal_loc, 'THAL')	# x_tick_locs[0]-(timeBuffer*1.3)

			plt.ylabel('')
			plt.yticks([])

		elif layerLines == 'layers':
			L1 = [0,100]			#	'NGF1'
			L2 = [0,0]				#  'IT2'  --> 'NGF2'
			L3 = [0,0] 				#  'IT3'  ---> 'NGF3'
			L4 = [0,0]				#	'ITP4' --> 'NGF4'
			L5A = [0,0]				#	'IT5A' --> 'NGF5A'
			L5B = [0,0] 			#  'IT5B'  --> 'NGF5B
			L6 =  [0,0]				#  'IT6' --> 'NGF6'
			Thal = [12000, 12907]	#	'TC' --> 'TIM'

			listL1 = []
			listL2_TOP = []
			listL2_BOTTOM = []
			listL3_TOP = []
			listL3_BOTTOM = []
			listL4_TOP = []
			listL4_BOTTOM = []
			listL5A_TOP = []
			listL5A_BOTTOM = []
			listL5B_TOP = []
			listL5B_BOTTOM = []
			listL6_TOP = []
			listL6_BOTTOM = []
			listThal_TOP = []
			listThal_BOTTOM = []

			for spk in spkInds:
				if indPop[spk] == 'NGF1':
					listL1.append(spk)

				elif indPop[spk] == 'IT2':
					listL2_TOP.append(spk)
				elif indPop[spk] == 'NGF2':
					listL2_BOTTOM.append(spk)

				elif indPop[spk] == 'IT3':
					listL3_TOP.append(spk)
				elif indPop[spk] == 'NGF3':
					listL3_BOTTOM.append(spk)

				elif indPop[spk] == 'ITP4':
					listL4_TOP.append(spk)
				elif indPop[spk] == 'NGF4':
					listL4_BOTTOM.append(spk)

				elif indPop[spk] == 'IT5A':
					listL5A_TOP.append(spk)
				elif indPop[spk] == 'NGF5A':
					listL5A_BOTTOM.append(spk)

				elif indPop[spk] == 'IT5B':
					listL5B_TOP.append(spk)
				elif indPop[spk] == 'NGF5B':
					listL5B_BOTTOM.append(spk)

				elif indPop[spk] == 'IT6':
					listL6_TOP.append(spk)
				elif indPop[spk] == 'NGF6':
					listL6_BOTTOM.append(spk)

				elif indPop[spk] == 'TC':
					listThal_TOP.append(spk)
				elif indPop[spk] == 'TIM':
					listThal_BOTTOM.append(spk)


			L1[0] = min(listL1)
			L1[1] = max(listL1)
			plt.axhline(0, color='black', linestyle=":")

			if len(listL2_TOP) > 0: 	 # NECESSARY FOR ALPHA OSC EVENT 
				L2[0] = min(listL2_TOP)
			else:
				L2[0] = max(listL1)
			L2[1] = max(listL2_BOTTOM)
			plt.axhline(L2[0], color='black', linestyle=":")

			L3[0] = min(listL3_TOP)
			L3[1] = max(listL3_BOTTOM)
			plt.axhline(L3[0], color='black', linestyle=":")

			L4[0] = min(listL4_TOP)
			L4[1] = max(listL4_BOTTOM)
			plt.axhline(L4[0], color='black', linestyle=":")

			L5A[0] = min(listL5A_TOP)
			L5A[1] = max(listL5A_BOTTOM)
			plt.axhline(L5A[0], color='black', linestyle=":")
			
			L5B[0] = min(listL5B_TOP)
			L5B[1] = max(listL5B_BOTTOM)
			plt.axhline(L5B[0], color='black', linestyle=":")

			if len(listL6_TOP) > 0:  ## NECESSARY FOR BETA OSC EVENT 
				L6[0] = min(listL6_TOP)
			else:
				L6[0] = max(listL5B_BOTTOM)
			L6[1] = max(listL6_BOTTOM)
			plt.axhline(L6[0], color='black', linestyle=":")

			Thal[0] = min(listThal_TOP)
			Thal[1] = max(listThal_BOTTOM)
			plt.axhline(Thal[0], color='black', linestyle=":")


			## Annotate y-axis with layer region labels ## 
			text_loc = waveletInfo[band]['layer_loc']
			print('layer location, ' + band + ': ' + str(text_loc))

			L1_loc = ((0 + L2[0])/2)*1.5
			plt.text(text_loc, L1_loc, 'L1') 		# x_tick_locs[0]-(timeBuffer*1.2)

			L2_loc = ((L2[0] + L3[0])/2)*1.5
			plt.text(text_loc, L2_loc, 'L2')
			
			L3_loc = (L3[0] + L4[0])/2
			plt.text(text_loc, L3_loc, 'L3')

			L4_loc = (L4[0] + L5A[0])/2
			plt.text(text_loc, L4_loc, 'L4')

			L5A_loc = ((L5A[0] + L5B[0])/2)*1.025
			plt.text(text_loc, L5A_loc, 'L5A')

			L5B_loc = (L5B[0] + L6[0])/2
			plt.text(text_loc, L5B_loc, 'L5B')

			L6_loc = (L6[0] + Thal[0])/2
			plt.text(text_loc, L6_loc, 'L6')

			thal_loc = ((Thal[0] + Thal[1]) / 2)*1.025
			plt.text(text_loc, thal_loc, 'THAL')

			plt.ylabel('')
			plt.yticks([])


		else: 
			## Setting y-axis labels ## 
			plt.ylabel('Neuron ID', fontsize=12) # Neurons (ordered by NCD within each pop)')
			plt.yticks(fontsize=10)


		## Print out plotting confirmed line 
		print(band + ' raster: PLOTTED')


		## SAVING ##
		if saveFigs:
			dataFileSplit = waveletInfo[band]['dataFile2'].split('.pkl')[0]
			rasterFilename = band + '_' + dataFileSplit
			if layerLines:
				rasterFile = savedir + rasterFilename + '_' + layerLines + '_RASTER.png'
			else:
				rasterFile = savedir + rasterFilename + '_RASTER.png'
			plt.savefig(rasterFile, dpi=300)
			print('RASTER SAVED')
		else:
			plt.show()



		# ### USING SIM.PLOTTING.PLOTRASTER ###
		# rasterData = sim.analysis.prepareRaster(timeRange=timeRangeBuffered, maxSpikes=1e8, orderBy=['pop'],popRates=True)

		# plt.rcParams={'figsize':(6.4, 4.8), 'dpi': 600}
		# print(plt.rcParams)


		# sim.plotting.plotRaster(rasterData=rasterData, orderBy=['pop'], popRates=False,
		# 						timeRange=timeRangeBuffered, orderInverse=True, marker=markerType, 
		# 						markerSize=12, ylabel='Neuron ID', legend=False, showFig=1, rcParams=plt.rcParams) # figSize=(6.4, 4.8), 

		# # import matplotlib; print(matplotlib.rcParams)
		# # plt.xlabel('Time (ms)', fontsize=12)
		# # plt.ylabel('Neuron ID', fontsize=12) # Neurons (ordered by NCD within each pop)')

		# # plt.xticks(fontsize=10)
		# # plt.yticks(fontsize=10)

		# # plt.title(band + ', Oscillation Event Raster', fontsize=14)

		# # ### f = plt.figure()







