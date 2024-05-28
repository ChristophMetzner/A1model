from simdat import *
import os

# basedir = '/g100_scratch/userexternal/egriffit/A1/v36_batch_eegSpeech_CINECA_trial_12/' # CINECA
# basedir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/speech/BBN_CINECA_speech_BEZ2018/' # CINECA
basedir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/BBN/BBN_CINECA_ANmodel/' # CINECA
# basedir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/speech/v34_batch_eegSpeech_CINECA_trial_1/'

allFiles = os.listdir(basedir)

pklFiles = []
for file in allFiles:
	if '_data.pkl' in file:
		pklFiles.append(file)


##### FOR INDIVIDUAL TESTING #####
# pklFiles = ['speechEEG_1_data.pkl']#['v36_batch_eegSpeech_CINECA_trial_12_1_data.pkl']#, 'v36_batch_eegSpeech_CINECA_trial_12_0_data.pkl']



#########################
#### SAVING TEST 2 ######
#########################
saveTest2 = 0

if saveTest2:
	for fn in pklFiles:
		## LOAD DATA 
		fullFilename = basedir + fn
		print('Working with file: ' + fn)
		simConfig, sdat, dstartidx, dendidx, dnumc, dspkID, dspkT = loaddat(fullFilename)
		print('loaddat run on ' + fn)

		#########################################
		#### SAVE ALL NON-CELLDIPOLES DATA! ##### 
		#########################################
		from scipy import io

		# get lidx, lty, and cellPos
		lidx = list(sdat['dipoleCells'].keys())
		lty = [GetCellType(idx,dnumc,dstartidx,dendidx) for idx in lidx]
		cellPos = [GetCellCoords(simConfig,idx) for idx in lidx]

		# create matDat dictionary
		matDat = {'cellPos': cellPos, 'cellPops': lty, 'lidx': lidx, 'dipoleSum': sdat['dipoleSum']}  # 'cellDipoles': cellDipoles, 

		# save matDat dictionary as a .mat file
		outfn_dir = basedir + 'trial_12_1_dipoleMatFiles/'
		outfn_matDat = outfn_dir + fn.split('_data.pkl')[0] + '_matDat.mat'  # '__' + partName + '__' + 'cellDipoles.mat' # outfn_matDat = fullFilename.split('_data.pkl')[0] + '_matDat.mat'  ### EDIT THIS TO GO TO PROPER DIRECTORY! 
		
		if not os.path.isfile(outfn_matDat):
			print('saving ' + outfn_matDat)
			io.savemat(outfn_matDat, matDat, do_compression=True)
			print('non-cellDipoles data saved!')
		else:
			print('matDat file already saved!')


		#############################
		### SAVE CELLDIPOLES DATA ### 
		#############################
		# get cellDipoles data 
		cellDipoles = [sdat['dipoleCells'][idx] for idx in lidx]


		# break lidx up into increments
		lidxLen = len(lidx)
		increment = 200 #25			# 20
		lidxIncrement = int(lidxLen / increment)  # /20 worked too!! 
		print('Separating lidx into ' + str(increment) + ' increments')

		import numpy as np
		lidxIncrements = (np.arange(0, lidxLen, lidxIncrement)).tolist()
		if lidxIncrements[-1] != lidxLen:
			lidxIncrements.append(lidxLen)


		# create cellDipolesDict w/ cellDipoles data + corresponding lidx indices 
		cellDipolesDict = {}
		for i in range(len(lidxIncrements)-1):
			keyName = 'part' + str(i)
			cellDipolesDict[keyName] = {'cellDipoles': cellDipoles[lidxIncrements[i]:lidxIncrements[i+1]], 
										'lidx': lidx[lidxIncrements[i]:lidxIncrements[i+1]]}
		print('cellDipolesDict completed -- lidx & cellDipoles data included')



		# SAVE cellDipoles dictionary to separate matlab files 
		outfn_dir = basedir + 'dipoleMatFiles/'
		if not os.path.exists(outfn_dir):
			os.mkdir(outfn_dir)

		for keyName in cellDipolesDict.keys():
			outfn = outfn_dir + fn.split('_data.pkl')[0] + '__' + keyName + '__' + 'cellDipoles.mat'
			if not os.path.isfile(outfn):
				print('saving ' + outfn)
				io.savemat(outfn, cellDipolesDict[keyName], do_compression=True)
				print(outfn + ' SAVED!')
			else:
				print('already saved!')





#######################
#### SAVING TEST ######
#######################
saveTest = 0

if saveTest:
	for fn in pklFiles:
		## LOAD DATA 
		fullFilename = basedir + fn
		print('Working with file: ' + fn)
		simConfig, sdat, dstartidx, dendidx, dnumc, dspkID, dspkT = loaddat(fullFilename)
		print('loaddat run on ' + fn)



		#########################################
		## BREAKING UP INTO SMALLER INCREMENTS ##
		from scipy import io
		lidx = list(sdat['dipoleCells'].keys())

		lidxLen = len(lidx)
		increment = 20
		lidxIncrement = int(lidxLen / increment)  # /15 worked too!! 

		import numpy as np
		lidxIncrements = (np.arange(0, lidxLen, lidxIncrement)).tolist()
		if lidxIncrements[-1] != lidxLen:
			lidxIncrements.append(lidxLen)

		# Want to create --> 
		## lidx_part0 = lidx[lidxIncrements[0]:lidxIncrements[1]]
		## lidx_part1 = lidx[lidxIncrements[1]:lidxIncrements[2]]

		lidxDict = {}
		for i in range(len(lidxIncrements)-1):
			keyName = 'lidx_part' + str(i)
			lidxDict[keyName] = lidx[lidxIncrements[i]:lidxIncrements[i+1]]

		print('ldxDict completed')

		##############################
		###  ---  NOW DO LTY ----  ###
		##############################
		lty = [GetCellType(idx,dnumc,dstartidx,dendidx) for idx in lidx]

		# Want to create --> 
		## lty_part0 = lty[lidxIncrements[0]:lidxIncrements[1]]
		## lty_part1 = lty[lidxIncrements[1]:lidxIncrements[2]]

		ltyDict = {}
		for i in range(len(lidxIncrements)-1):
			keyName = 'lty_part' + str(i)
			ltyDict[keyName] = lty[lidxIncrements[i]:lidxIncrements[i+1]]

		print('ltyDict completed')

		##############################
		### --- NOW DO cellPos --- ###
		##############################

		cellPos = [GetCellCoords(simConfig,idx) for idx in lidx]

		cellPosDict = {}
		for i in range(len(lidxIncrements)-1):
			keyName = 'cellPos_part' + str(i)
			cellPosDict[keyName] = cellPos[lidxIncrements[i]:lidxIncrements[i+1]]

		print('cellPosDict completed')


		#################################
		### --- NOW DO cellDipoles--- ###
		#################################

		cellDipoles = [sdat['dipoleCells'][idx] for idx in lidx]

		cellDipolesDict = {}
		for i in range(len(lidxIncrements)-1):
			keyName = 'cellDipoles_part' + str(i)
			cellDipolesDict[keyName] = cellDipoles[lidxIncrements[i]:lidxIncrements[i+1]]

		print('cellDipolesDict completed')

		###########################
		### --- NOW DO matDat-- ###
		###########################

		matDatDict = {}

		for i in range(len(cellDipolesDict.keys())):
			keyName = 'matDat_part' + str(i)
			partKeyName = '_part' + str(i)
			matDatDict[keyName] = {'cellPos': cellPosDict['cellPos' + partKeyName], 
								'cellPops': ltyDict['lty' + partKeyName], 
								'cellDipoles': cellDipolesDict['cellDipoles' + partKeyName], 
								'dipoleSum': sdat['dipoleSum']}

		print('matDatDict completed')


		###########################
		### --- NOW SAVE !!! -- ###
		###########################

		for matDatPart in matDatDict.keys():
			partName = matDatPart.split('matDat_')[1] 

			outfn_dir = basedir + 'dipoleMatFiles/'
			if not os.path.exists(outfn_dir):
				os.mkdir(outfn_dir)

			outfn = outfn_dir + fn.split('_data.pkl')[0] + '__' + partName + '__' + 'dipoleMat.mat'   		#fullFilename.split('_data.pkl')[0] + '__' + partName + '__' + 'dipoleMat.mat'
			
			if not os.path.isfile(outfn):
				print('saving ' + outfn)
				io.savemat(outfn, matDatDict[matDatPart], do_compression=True)
				print(outfn + ' SAVED!')
			else:
				print('already saved!')



		#######
		###### FINISH CONVERTING THE REST OF THIS IN A SECOND ### 
		# print('lidx_part1: lidx[0:' + str(partialInd) + ']')
		# lidx_part1 = lidx[:partialInd]

		# lty = [GetCellType(idx,dnumc,dstartidx,dendidx) for idx in lidx]
		# lty_part1 = lty[:partialInd]


		# cellPos = [GetCellCoords(simConfig,idx) for idx in lidx]
		# cellPos_part1 = cellPos[:partialInd]

		# cellDipoles = [sdat['dipoleCells'][idx] for idx in lidx]
		# cellDipoles_part1 = cellDipoles[:partialInd]

		# matDat_part1 = {'cellPos': cellPos_part1, 'cellPops': lty_part1, 'cellDipoles': cellDipoles_part1, 'dipoleSum': sdat['dipoleSum']}

		# outfn_part1 = fullFilename.split('_data.pkl')[0] + '__PART_1_' + '_dipoleMat.mat'
		# io.savemat(outfn_part1, matDat_part1, do_compression=True)
		# print('part 1 saved')



#######################
#### DIPOLE TEST ######
#######################
dipoleMat = 0
if dipoleMat:

	for fn in pklFiles:
		## LOAD DATA 
		fullFilename = basedir + fn
		print('Working with file: ' + fn)
		simConfig, sdat, dstartidx, dendidx, dnumc, dspkID, dspkT = loaddat(fullFilename)
		print('loaddat run on ' + fn)

		
		# outfn = fullFilename.split('_data.pkl')[0] + '_dipoleMat.mat'
		# save_dipoles_matlab(outfn, simConfig, sdat, dnumc, dstartidx, dendidx)


		#####################
		## GO LINE BY LINE ##
		### NOTE: the strategy here was to do the original lines (like save_dipoles_matlab func from simdat.py, but then break the 
		### data into two smaller chunks for saving. ALTERNATIVE approach would be to break lidx into two parts as we do here
		### then use those two parts to guide the situation from there on out. I think what we have now is better, but just a note.)
		from scipy import io

		## determine lidx and then break lidx into two parts to make this smaller ## 
		lidx = list(sdat['dipoleCells'].keys())

		lidxLen = len(lidx)
		partialInd = int(lidxLen / 2)

		print('lidx_part1: lidx[0:' + str(partialInd) + ']')
		lidx_part1 = lidx[:partialInd]
		print('lidx_part2: lidx[' + str(partialInd) + ':]')
		lidx_part2 = lidx[partialInd:]


		## determine lty and also break this into two parts to make it smaller ## 
		lty = [GetCellType(idx,dnumc,dstartidx,dendidx) for idx in lidx]

		lty_part1 = lty[:partialInd]
		lty_part2 = lty[partialInd:]

		## determine cellPos and also break it into two parts ## 
		cellPos = [GetCellCoords(simConfig,idx) for idx in lidx]

		cellPos_part1 = cellPos[:partialInd]
		cellPos_part2 = cellPos[partialInd:]


		## determine cellDipoles and also break it into two parts ## 
		cellDipoles = [sdat['dipoleCells'][idx] for idx in lidx]

		cellDipoles_part1 = cellDipoles[:partialInd]
		cellDipoles_part2 = cellDipoles[partialInd:]


		## create dict w/ matlab data to save ## 
		matDat = {'cellPos': cellPos, 'cellPops': lty, 'cellDipoles': cellDipoles, 'dipoleSum': sdat['dipoleSum']}
		
		#### NOTE: not sure what to do about dipoleSum... hmm....
		matDat_part1 = {'cellPos': cellPos_part1, 'cellPops': lty_part1, 'cellDipoles': cellDipoles_part1, 'dipoleSum': sdat['dipoleSum']}
		matDat_part2 = {'cellPos': cellPos_part2, 'cellPops': lty_part2, 'cellDipoles': cellDipoles_part2, 'dipoleSum': sdat['dipoleSum']}


		## SAVE ## 
		outfn_part1 = fullFilename.split('_data.pkl')[0] + '_PART_1_' + '_dipoleMat.mat'
		outfn_part2 = fullFilename.split('_data.pkl')[0] + '_PART_2_' + '_dipoleMat.mat'
		io.savemat(outfn_part1, matDat_part1, do_compression=True)
		print('part 1 saved')
		io.savemat(outfn_part2, matDat_part2, do_compression=True)
		print('part 2 saved')


		############################################################################
		#### ORIGINAL ##### 
		# from scipy import io

		# lidx = list(sdat['dipoleCells'].keys())
		# lty = [GetCellType(idx,dnumc,dstartidx,dendidx) for idx in lidx]

		# cellPos = [GetCellCoords(simConfig,idx) for idx in lidx]
		# cellDipoles = [sdat['dipoleCells'][idx] for idx in lidx]

		# matDat = {'cellPos': cellPos, 'cellPops': lty, 'cellDipoles': cellDipoles, 'dipoleSum': sdat['dipoleSum']}


		### print('dipoles saved to matlab file!')


###########################
#### RASTER PLOTTING ######
###########################
raster = 1

if raster:

	for fn in pklFiles:
		fullFilename = basedir + fn

		print('Working with file: ' + fn)

		simConfig, sdat, dstartidx, dendidx, dnumc, dspkID, dspkT = loaddat(fullFilename)

		print('loaddat run on ' + fn)

		rasterFilename = fullFilename.split('_data.pkl')[0] + '_raster.png'
		drawraster(dspkT,dspkID,dnumc,tlim=None,msz=0.5,skipstim=False,drawlegend=True,saveFig=True,rasterFile=rasterFilename) # skipstim=True
		## ^^^ HOW TO SAVE?? 

		print('raster for file ' + fn + ' has been drawn!')




