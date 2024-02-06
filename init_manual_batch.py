"""
init.py

Starting script to run multi-seed batch of the NetPyNE-based A1 model.



Contributors: ericaygriffith@gmail.com, salvadordura@gmail.com, christoph.metzner@gmail.com
"""

import matplotlib; matplotlib.use('Agg')  # to avoid graphics error in servers

from netpyne import sim



def run(seed,label):
	cfg, netParams = sim.readCmdLineArgs(simConfigDefault='cfg.py', netParamsDefault='netParams_addFB_alterSyn.py')

	cfg.seeds = {'conn': seed, 'stim': 1, 'loc': 1}
	cfg.simLabel = 'label'
	# sim.createSimulateAnalyze(netParams, cfg)
	print(cfg.simLabel)
	sim.initialize(simConfig = cfg, netParams = netParams)  				# create network object and set cfg and net params
	sim.net.createPops()               			# instantiate network populations
	sim.net.createCells()              			# instantiate network cells based on defined populations
	sim.net.connectCells()            			# create connections between cells based on params
	sim.net.addStims() 							# add network stimulation
	sim.setupRecording()              			# setup variables to record for each cell (spikes, V traces, etc)
	sim.runSim()                      			# run parallel Neuron simulation  

	# distributed saving (to avoid errors with large output data)
	sim.saveDataInNodes()
	sim.gatherDataFromFiles()
	
	sim.saveData()  

def main():
	args = sys.argv[1:]
	
	run(args[0],args[1])
	
if __name__ == '__main__':
	main()
