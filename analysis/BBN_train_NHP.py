"""
BBN_train_NHP.py 

Code to compare PSD of LFP data in model and NHP in response to low-frequency BBN stimulus trains

Contributors: ericaygriffith@gmail.com, samnemo@gmail.com
"""

from expDataAnalysis import *



###########################
######## MAIN CODE ########
###########################

if __name__ == '__main__':

    # Parent data directory containing .mat files
    origDataDir = '../data/NHPdata/BBN/' 


    recordingArea = 'A1/' 

    testFiles = ['1-rb067068029@os.mat']

    if test:
        dataFiles = testFiles
    else:
        dataPath = origDataDir + recordingArea
        dataFilesList = os.listdir(dataPath) 
        dataFiles = []
        for file in dataFilesList:
            if '.mat' in file:
                dataFiles.append(file)

    # setup netpyne
    samprate = 11*1e3  # in Hz
    sim.initialize()
    sim.cfg.recordStep = 1000./samprate # in ms


    for dataFile in dataFiles: 
        
        fullPath = origDataDir + dataFile     # Path to data file 

        [sampr,LFP_data,dt,tt,CSD_data,trigtimes] = loadfile(fn=fullPath, samprds=samprate, spacing_um=100)
                # sampr is the sampling rate after downsampling 
                # tt is time array (in seconds)
                # trigtimes is array of stim trigger indices
                # NOTE: make samprds and spacing_um args in this function as well for increased accessibility??? 

        ##### SET PATH TO .csv LAYER FILE ##### 
        dbpath = '../data/NHPdata/BBN/19jun21_A1_ERP_Layers.csv' 

        ##### GET LAYERS FOR OVERLAY #####
        s1low,s1high,s2low,s2high,glow,ghigh,i1low,i1high,i2low,i2high = getflayers(fullPath,dbpath=dbpath,getmid=False,abbrev=False) # fullPath is to data file, dbpath is to .csv layers file 
        lchan = {}
        lchan['S'] = s2high
        lchan['G'] = ghigh
        lchan['I'] = CSD_data.shape[0]-1 #i2high
        print('s2high: ' + str(s2high))


        electrodes = [1, 11, 13] 
        
        # plot LFP PSDs
        ## (1) PRIOR TO STIMULUS ONSET
        allData_prior = plotLFP(dat=LFP_data, tt=tt, timeRange=[0,10000], plots=['PSD'], electrodes=electrodes, minFreq=0.1, maxFreq=3, stepFreq=0.1, normSignal=False, normPSD=False, saveFig=True, fn=fullPath, dbpath=dbpath) 

        ## (2) DURING STIMULUS TRAIN 
        allData_during = plotLFP(dat=LFP_data, tt=tt, timeRange=[15000,85000], plots=['PSD'], electrodes=electrodes, minFreq=0.1, maxFreq=3, stepFreq=0.1, normSignal=False, normPSD=False, saveFig=True, fn=fullPath, dbpath=dbpath) 




