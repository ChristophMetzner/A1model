"""
batch.py 

Batch simulation for M1 model using NetPyNE

Contributors: salvadordura@gmail.com
"""
from netpyne.batch import Batch
from netpyne import specs
import numpy as np


# ----------------------------------------------------------------------------------------------
# 40 Hz ASSR optimization
# ----------------------------------------------------------------------------------------------
def assr_batch(filename):
    params = specs.ODict()

    if not filename:
        filename = 'data/v34_batch25/trial_2142/trial_2142_cfg.json'

    # from prev 
    import json
    with open(filename, 'rb') as f:
        cfgLoad = json.load(f)['simConfig']
    cfgLoad2 = cfgLoad


    minF_E = 1.25 
    maxF_E = 1.5

    minF_I = 0.1 
    maxF_I = 0.25


    # #### SET CONN AND STIM SEEDS #### 
    params[('L4L3PV ')] = [minF_I,maxF_]
    params[('L4L3SOM ')] = [minF_I,maxF_I]
    params[('L4L3E ')] = [minF_E,maxF_]
    #### GROUPED PARAMS #### 
    groupedParams = [] 

    # --------------------------------------------------------
    # initial config
    initCfg = {} # set default options from prev sim
    
    initCfg['duration'] = 6000 #11500 
    initCfg['printPopAvgRates'] = [1500, 2500]
    initCfg['scaleDensity'] = 1.0 
    initCfg['recordStep'] = 0.05

    # SET SEEDS FOR CONN AND STIM 
    initCfg[('seeds', 'conn')] = 0




    ### OPTION TO RECORD EEG / DIPOLE ###
    initCfg['recordDipole'] = True
    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False
    
    # from prev - best of 50% cell density
    updateParams = ['EEGain', 'EIGain', 'IEGain', 'IIGain',
                    ('EICellTypeGain', 'PV'), ('EICellTypeGain', 'SOM'), ('EICellTypeGain', 'VIP'), ('EICellTypeGain', 'NGF'),
                    ('IECellTypeGain', 'PV'), ('IECellTypeGain', 'SOM'), ('IECellTypeGain', 'VIP'), ('IECellTypeGain', 'NGF'),
                    ('EILayerGain', '1'), ('IILayerGain', '1'),
                    ('EELayerGain', '2'), ('EILayerGain', '2'),  ('IELayerGain', '2'), ('IILayerGain', '2'), 
                    ('EELayerGain', '3'), ('EILayerGain', '3'), ('IELayerGain', '3'), ('IILayerGain', '3'), 
                    ('EELayerGain', '4'), ('EILayerGain', '4'), ('IELayerGain', '4'), ('IILayerGain', '4'), 
                    ('EELayerGain', '5A'), ('EILayerGain', '5A'), ('IELayerGain', '5A'), ('IILayerGain', '5A'), 
                    ('EELayerGain', '5B'), ('EILayerGain', '5B'), ('IELayerGain', '5B'), ('IILayerGain', '5B'), 
                    ('EELayerGain', '6'), ('EILayerGain', '6'), ('IELayerGain', '6'), ('IILayerGain', '6')] 

    for p in updateParams:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad[p]})

    # good thal params for 100% cell density 
    updateParams2 = ['thalamoCorticalGain', 'intraThalamicGain', 'EbkgThalamicGain', 'IbkgThalamicGain', 'wmat']

    for p in updateParams2:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad2[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad2[p]})



    # --------------------------------------------------------
    # fitness function
    fitnessFuncArgs = {}

    fitnessFuncArgs['maxFitness'] = 2000

    def fitnessFunc(simData, **kwargs):
        import numpy as np
        from scipy import signal as ss
        
        fs = 10000
        nperseg = int(fs/2)
        #s = int(1.75*fs)
        s = int(0) 

        electrodes = [3,4,5,6,7,8,9,10,11,12]
        powers = np.zeros((len(electrodes),))

        for i,e in enumerate(electrodes):
            lfps = np.array(simData['LFP'])
            lfp = lfps[s:,e]
            freq_wel, ps_wel = ss.welch(lfp,fs=fs,nperseg=nperseg)
            powers[i] = np.sum(ps_wel[38:42])
        
        fitness = (10**4)*np.mean(powers)

        info = '; '.join(['power=%.6f fit=%.2f' % (p, fitness) for p in powers])
        print('  ' + info)

        return fitness


    b = Batch(params=params, netParamsFile='netParams_ASSR.py', cfgFile='cfg_ASSR.py', initCfg=initCfg, groupedParams=groupedParams)
    
    b.method = 'optuna'
    
    b.optimCfg = {
        'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
        'fitnessFuncArgs': fitnessFuncArgs,
        'maxFitness': fitnessFuncArgs['maxFitness'],
        'maxiters':     2,    #    Maximum number of iterations (1 iteration = 1 function evaluation)
        'maxtime':      None,    #    Maximum time allowed, in seconds
        'maxiter_wait': 500,
        'time_sleep': 150,
        'popsize': 1  # unused - run with mpi 
    }

    return b



def assr_batch_grid(filename):
    params = specs.ODict()

    if not filename:
        filename = 'data/v34_batch25/trial_2142/trial_2142_cfg.json'

    # from prev 
    import json
    with open(filename, 'rb') as f:
        cfgLoad = json.load(f)['simConfig']
    cfgLoad2 = cfgLoad

    # #### SET weights#### 
    params[('thalL4VIP')] = [0.05, 0.1]
    params[('thalL4NGF')] = [0.05, 0.1]
    params[('L4L3VIP')] = [0.05, 0.1]


    #### GROUPED PARAMS #### 
    groupedParams = [] 

    # --------------------------------------------------------
    # initial config
    initCfg = {} # set default options from prev sim
    
    initCfg['duration'] = 6000 #11500 
    initCfg['printPopAvgRates'] = [1500, 3500]
    initCfg['scaleDensity'] = 1.0 
    initCfg['recordStep'] = 0.05

    # SET SEEDS FOR CONN AND STIM 
    initCfg[('seeds', 'conn')] = 0




    ### OPTION TO RECORD EEG / DIPOLE ###
    initCfg['recordDipole'] = True
    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False
    
    # from prev - best of 50% cell density
    updateParams = ['EEGain', 'EIGain', 'IEGain', 'IIGain',
                    ('EICellTypeGain', 'PV'), ('EICellTypeGain', 'SOM'), ('EICellTypeGain', 'VIP'), ('EICellTypeGain', 'NGF'),
                    ('IECellTypeGain', 'PV'), ('IECellTypeGain', 'SOM'), ('IECellTypeGain', 'VIP'), ('IECellTypeGain', 'NGF'),
                    ('EILayerGain', '1'), ('IILayerGain', '1'),
                    ('EELayerGain', '2'), ('EILayerGain', '2'),  ('IELayerGain', '2'), ('IILayerGain', '2'), 
                    ('EELayerGain', '3'), ('EILayerGain', '3'), ('IELayerGain', '3'), ('IILayerGain', '3'), 
                    ('EELayerGain', '4'), ('EILayerGain', '4'), ('IELayerGain', '4'), ('IILayerGain', '4'), 
                    ('EELayerGain', '5A'), ('EILayerGain', '5A'), ('IELayerGain', '5A'), ('IILayerGain', '5A'), 
                    ('EELayerGain', '5B'), ('EILayerGain', '5B'), ('IELayerGain', '5B'), ('IILayerGain', '5B'), 
                    ('EELayerGain', '6'), ('EILayerGain', '6'), ('IELayerGain', '6'), ('IILayerGain', '6')] 

    for p in updateParams:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad[p]})

    # good thal params for 100% cell density 
    updateParams2 = ['thalamoCorticalGain', 'intraThalamicGain', 'EbkgThalamicGain', 'IbkgThalamicGain', 'wmat']

    for p in updateParams2:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad2[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad2[p]})



    b = Batch(params=params, netParamsFile='netParams_ASSR.py', cfgFile='cfg_ASSR.py', initCfg=initCfg, groupedParams=groupedParams)
    b.method = 'grid'

    return b



# ----------------------------------------------------------------------------------------------
# Run configurations
# ----------------------------------------------------------------------------------------------
def setRunCfg(b, type='mpi_direct'):
    if type=='mpi_direct':
        b.runCfg = {'type': 'mpi_direct',
            'cores': 8,
            'script': 'init.py',
            'mpiCommand': 'mpiexec',
            'skip': True}

# ----------------------------------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------------------------------

if __name__ == '__main__':



    #b = assr_batch('data/v34_batch25/trial_2142/trial_2142_cfg.json')
    b = assr_batch_grid('data/v34_batch25/trial_2142/trial_2142_cfg.json')

    b.batchLabel = 'ASSR_grid_09_01_24_lesath'   
    b.saveFolder = 'data/'+b.batchLabel

    setRunCfg(b, 'mpi_direct')
    b.run() # run batch



    
