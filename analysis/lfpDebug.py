from netpyne import sim
import numpy as np
import matplotlib.pyplot as plt 
from neuron import h


### Function to load sim from .pkl file:
def loadFile(batchLabel, gridNumber): 				#if fullPath is None: # Change the args up alittle -- confusing 
	if gridNumber is None:    
		fn = batchLabel + '_data.pkl'
		fullPath = '../data/' + batchLabel + '/' + fn
	else:
		fn = batchLabel + '_' + gridNumber + '_data.pkl'
		fullPath = '../data/' + batchLabel + '/' + gridNumber + '/' + fn
	sim.load(fullPath, instantiate=True) # instantiate = False gives empty sim.net.compartCells


### Plot LFP 
def plotLFP(plotTypes, timeRange, electrodes):
	if timeRange is None:
		timeRange = [0, sim.cfg.duration]
	else:
		timeRange = timeRange

	if electrodes is None:
		electrodes = [3, 10, 13, 18]
	else:
		electrodes = electrodes

	if plotTypes is None:
		plotTypes = ['timeSeries']
	else:
		plotTypes = plotTypes

	sim.analysis.plotLFP(plots=plotTypes, timeRange=timeRange, electrodes=electrodes)


### Figure out where LFP signal is going to zero 
def zeroesLFP(timeRange, electrode):
	if timeRange is None:
		timeRange = [0,sim.cfg.duration]
	#t = np.arange(timeRange[0], timeRange[1], sim.cfg.recordStep)
	
	if electrode is None:
		elec = 10
	else:
		elec = electrode

	lfp = np.array(sim.allSimData['LFP'])[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]
	lfpElec = lfp[:, elec] 

	lfpZeroInd = np.where(lfpElec==0)
	lfpZeroInd = list(lfpZeroInd[0])
	return lfpZeroInd







#### Main code block:


# load sim file 
batchLabel = 'v34_batch27_0_3_QD_currentRecord2'   #fn = '../data/v34_batch27_0_3_QD_currentRecord1/v34_batch27_0_3_QD_currentRecord1_data.pkl'
loadFile(batchLabel=batchLabel, gridNumber='1_1')


# print data types stored in sim.allSimData 
allData = sim.allSimData
print(allData.keys())



## Plot LFP 
plotLFP(plotTypes=['timeSeries'], timeRange=None, electrodes=[5, 10])
# plt.plot(t[0:len(lfpPlot)], lfpPlot, linewidth=1.0)
# plt.plot(t,lfp[:,elec]) # 10
# plt.plot(t, lfpPlot, linewidth=1.0)
# plt.show()


## Figure out which indices of lfp goes to zero to see how often this happens 
lfpZeroInd = zeroesLFP(timeRange=None, electrode=None)
print('number of times lfp goes to zero: ' + str(len(lfpZeroInd)))


# print('lfpZeroInd[0]: ' + str(lfpZeroInd[0]))
# print('lfpZeroInd[1]: ' + str(lfpZeroInd[1]))
# print('lfpZeroInd[2]: ' + str(lfpZeroInd[2]))
# print('lfpZeroInd[3]: ' + str(lfpZeroInd[3]))
# print('lfpZeroInd[4]: ' + str(lfpZeroInd[4]))
# print('lfpZeroInd[5]: ' + str(lfpZeroInd[5]))
# print('lfpZeroInd[6]: ' + str(lfpZeroInd[6]))
# print('lfpZeroInd[7]: ' + str(lfpZeroInd[7]))




###########################################
## Testing tr and im 
# cell0 = sim.net.compartCells[0]
# cell1 = sim.net.compartCells[1]

# gid0 = cell0.gid
# gid1 = cell1.gid

# im0 = cell0.getImemb()
# im1 = cell1.getImemb()

# print('im0: ' + str(im0))
# print('im1: ' + str(im1))

# #### look at all membrane currents at the end
# count = 0
# for i in range(len(sim.net.compartCells)):
# 	cell = sim.net.compartCells[i]
# 	im = cell.getImemb()
# 	if not list(im ==0):
# 		count += 1

# print('count: ' + str(count))

# tr0 = sim.net.recXElectrode.getTransferResistance(gid0)
# tr1 = sim.net.recXElectrode.getTransferResistance(gid1)

# print('tr0: ' + str(tr0))
# print('tr1: ' + str(tr1))

# ecp0 = np.dot(tr0, im0)
# ecp1 = np.dot(tr1, im1)

# print('ecp0: ' + str(ecp0))
# print('ecp1: ' + str(ecp1))


###########################################
# # Loading individual LFP traces
# LFPCellDict = sim.allSimData['LFPCells']
# print('Dict keys for LFPCells: ' + str(LFPCellDict.keys()))

# LFPCells = LFPCellDict.keys()
# for cell in LFPCells:
# 	elec = 4  	# arbitrary -- which electrode do you want to plot?
# 	LFPtrace = LFPCellDict[cell][:,elec]
# 	LFPtrace = list(LFPtrace) ## necessary?
# 	plt.plot(t,LFPtrace)
# 	plt.show()
#### COMMENTED OUT JUST NOW WHEN LOOKING AT MEMBRANE VOLTAGES


###########
## Look at membrane voltages 

# allData = sim.allSimData 
# print(allData.keys())
# membraneVoltage = allData['V_soma']
# cells = list(membraneVoltage.keys())

# #membraneVoltage[cells[0]]

# plt.plot(t, membraneVoltage[cells[0]])
# plt.show()

###########
## Look at membrane currents

# timeRange = [0,sim.cfg.duration]
# t = np.arange(timeRange[0], timeRange[1], sim.cfg.recordStep)

# membraneCurrent = allData['I_soma']
# cells = list(membraneCurrent.keys())
# print(len(cells))
# if len(membraneCurrent[0]) == len(t):
# 	print('TIME and I_soma[0] EQUAL')

# membraneCurrent0 = membraneCurrent[cells[0]]
# membraneCurrent1 = membraneCurrent[cells[1]]

# for cell in cells[40:-1]:
# 	membraneCurrentCell = membraneCurrent[cell]
# 	plt.plot(t, membraneCurrentCell[1:])


