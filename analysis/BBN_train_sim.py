"""
BBN_train_sim.py

Code to calculate and visualize PSD of LFP data in model in response to low-frequency BBN stimulus trains

Contributors: ericaygriffith@gmail.com 
"""

from netpyne import sim
from simDataAnalysis import *


def sim_PSD_LFP(dataFile,timeRange, maxFreq):
	simData = sim.load(dataFile, instantiate=False)
	plotLFP(timeRange=timeRange, plots=['PSD'], electrodes=[4, 11, 13], normSignal=False, normPSD=False, minFreq=0.1, maxFreq=maxFreq, stepFreq=0.1)


def slide3_model_vs_NHP():
	# 1.6 HZ, PRIOR TO STIM ONSET 
	sim_PSD_LFP(dataFile='../data/simDataFiles/BBN/BBN_CINECA_v36_5656BF_SOA624/BBN_CINECA_v36_5656BF_624ms_data.pkl', timeRange = [0, 2500], maxFreq=3)
	# 1.6 HZ, DURING STIM TRAIN
	sim_PSD_LFP(dataFile='../data/simDataFiles/BBN/BBN_CINECA_v36_5656BF_SOA624/BBN_CINECA_v36_5656BF_624ms_data.pkl', timeRange = [2500, 10000], maxFreq=3)

def slide7_wildtype():
	# 3.3 HZ, wildtype, prior to stim onset 
	sim_PSD_LFP(dataFile = '../data/simDataFiles/BBN/v39_BBN_E_to_I/v39_BBN_E_to_I_0_0_data.pkl', timeRange = [0,2500], maxFreq=5)
	# 3.3 HZ, wildtype, during stim train
	sim_PSD_LFP(dataFile = '../data/simDataFiles/BBN/v39_BBN_E_to_I/v39_BBN_E_to_I_0_0_data.pkl', timeRange = [2500,10000], maxFreq=5)


def slide8_NMDA():
	# 3.3 Hz, ALTERED NMDA E --> I, during stim train
	sim_PSD_LFP(dataFile = '../data/simDataFiles/BBN/v39_BBN_E_to_I/v39_BBN_E_to_I_0_1_data.pkl', timeRange = [2500,10000], maxFreq=5)




if __name__ == '__main__':
	# TO GENERATE FIGS IN SLIDE 3 - model vs NHP ## 
	slide3_model_vs_NHP()

	# TO GENERATE FIGS IN SLIDE 7 -- WILDTYPE MODEL in NMDA ALTERATIONS ## 
	slide7_wildtype()

	# TO GENERATE FIGS IN SLIDE 8 -- ALTERED NMDA MODEL ## 
	slide8_NMDA()

