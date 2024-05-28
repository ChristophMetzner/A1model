import json
#import pickle
import pickle5 as pickle
import os
import matplotlib.pyplot as plt 
import numpy as np

###################
#### FUNCTIONS ####
###################

##################################################
def topIndividualPops(topPopsData, freqBand, region):
	## OUTPUT: bar graph with the top represented population(s) for given osc event subtype
	#### topPopsData	: dict 
	#### freqBand 		: str
	#### region 		: str 

	eventIndices = list(topPopsData[freqBand][region].keys())

	popCounts = {}

	for eventIdx in eventIndices:
		topPops = topPopsData[freqBand][region][eventIdx]
		for pop in topPops:
			if pop in popCounts.keys():
				popCounts[pop] += 1 
			else:
				popCounts[pop] = 1



	popCountsSorted = {k: v for k, v in sorted(popCounts.items(), key=lambda item: item[1], reverse=True)}


	return popCountsSorted





#############################################
def topPopGroups(topPopsData, freqBand, region):
	## OUTPUT: bar graph with the most popular trios for given osc event subtype
	#### topPopsData	: dict 
	#### freqBand 		: str
	#### region 		: str 

	eventIndices = list(topPopsData[freqBand][region].keys())

	trioCounts = {}

	for eventIdx in eventIndices:
		trioSorted = sorted(topPopsData[freqBand][region][eventIdx])
		if str(trioSorted) not in trioCounts.keys():
			trioCounts[str(trioSorted)] = 1
		else:
			trioCounts[str(trioSorted)] += 1

	trioCountsSorted = {k: v for k, v in sorted(trioCounts.items(), key=lambda item: item[1], reverse=True)}


	return trioCountsSorted



def barPlot(countsDict, freqBand, region, fileName):
	## countsDict: dict -- e.g. popCounts or trioCounts 
	## freqBand: str (e.g. 'delta')
	## region: str (e.g. 'supra' , 'gran', 'infra') 

	pops = list(countsDict.keys())
	counts = []
	for pop in pops:
		counts.append(countsDict[pop])
	plt.bar(pops, counts)

	plt.xlabel('Populations')

	## Trying to figure out how to best label these for trioCounts 
	if len(pops[0]) > 15:
		xlabelTest = np.arange(len(pops))
		plt.xticks(xlabelTest, pops, rotation=15, fontsize=8) # rotation=15,

	# if region == 'supra':
	# 	regionFull = 'supragranular'
	plotTitle = freqBand + ', ' + region
	plt.title(plotTitle)

	if fileName:
		plt.savefig(fileName)
	plt.show()



################################################
# def popSourceSink(oscEventData,freqBand, region):
	## OUTPUT: Identify / Visualize the sources and sinks for given osc event subtype

	### oscEventData[freqBand][region][subject][eventIdx]['maxPops_avgCSD']['elec'][pop] = CSD value





################################################
############# MAIN CODE BLOCK ##################
################################################

# cortType = 'ICortPops'  
# cortType = 'ECortPops' 
cortType = 'AllCortPops'


if cortType == 'ICortPops':
	based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/topPops/ICortPops/'
elif cortType == 'ECortPops':
	based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/topPops/ECortPops/'
elif cortType == 'AllCortPops':
	based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/topPops/AllCortPops/'



os.chdir(based)

allFiles = os.listdir()

topPopsAvgFiles = []
topPopsPeakFiles = []
oscEventFiles = []

for file in allFiles:
	if 'topPopsAvg' in file:
		topPopsAvgFiles.append(file)
	if 'topPopsPeak' in file:
		topPopsPeakFiles.append(file)
	if 'oscEventInfo' in file:
		oscEventFiles.append(file)

# print(topPopsFiles)
# print(oscEventFiles)


## LOAD TOP POPS AVG DATA
topPopsAvgData = {}
for topPopAvgFile in topPopsAvgFiles:
	f = open(topPopAvgFile)
	data = json.load(f)
	topPopsAvgData.update(data)

## LOAD TOP POPS PEAK DATA
topPopsPeakData = {}
for topPopPeakFile in topPopsPeakFiles:
	f = open(topPopPeakFile)
	data = json.load(f)
	topPopsPeakData.update(data)


## LOAD OSC EVENT INFO INTO DICT 
oscEventData = {}
for oscEventFile in oscEventFiles:
	with open(oscEventFile, 'rb') as handle:
		data = pickle.load(handle)
	oscEventData.update(data)


### Synthesize findings from topPops Data ###
regions = ['supra', 'gran', 'infra']
freqBands = ['delta', 'theta', 'alpha', 'beta']


for freqBand in freqBands:
	for region in regions:
		# popCountsAvg = topIndividualPops(topPopsAvgData, freqBand, region)
		# trioCountsAvg = topPopGroups(topPopsAvgData, freqBand, region)

		popCountsPeak = topIndividualPops(topPopsPeakData, freqBand, region)
		trioCountsPeak = topPopGroups(topPopsPeakData, freqBand, region)

		fileNamePeak = freqBand + '_' + region + '_peak.png'
		barPlot(popCountsPeak, freqBand=freqBand, region=region, fileName=fileNamePeak)

########
# freqBand = 'alpha'
# region = 'infra'



# popCountsAvg = topIndividualPops(topPopsAvgData, freqBand, region)
# trioCountsAvg = topPopGroups(topPopsAvgData, freqBand, region)

# popCountsPeak = topIndividualPops(topPopsPeakData, freqBand, region)
# trioCountsPeak = topPopGroups(topPopsPeakData, freqBand, region)


#barPlot(popCountsAvg, freqBand='theta', region='supra')
#barPlot(popCountsPeak, freqBand=freqBand, region=region)
# barPlot(trioCountsAvg, freqBand='theta', region='supra')
#barPlot(trioCountsPeak, freqBand=freqBand, region=region)















######### POPULATIONS ######### 
# ECortPops = ['IT2', 
# 			 'IT3', 
# 			 'ITP4', 'ITS4', 
# 			 'IT5A', 'CT5A', 
# 			 'IT5B', 'CT5B', 'PT5B', 
# 			 'IT6', 'CT6']

# ICortPops = ['NGF1', 
# 			'PV2', 'SOM2', 'VIP2', 'NGF2', 
# 			'PV3', 'SOM3', 'VIP3', 'NGF3',
# 			'PV4', 'SOM4', 'VIP4', 'NGF4',
# 			'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',
# 			'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',
# 			'PV6', 'SOM6', 'VIP6', 'NGF6']

# AllCortPops = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4',
# 'PV4', 'SOM4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B', 'PV5B',
# 'SOM5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6']












