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


    minF = 0.1 
    maxF = 2.0


    # #### SET CONN AND STIM SEEDS #### 
    params[('thalL4PV ')] = [minF,maxF]
    params[('thalL4SOM ')] = [minF,maxF]
    params[('thalL4E ')] = [minF,maxF]
    #### GROUPED PARAMS #### 
    groupedParams = [] 

    # --------------------------------------------------------
    # initial config
    initCfg = {} # set default options from prev sim
    
    initCfg['duration'] = 2500 #11500 
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
        s = 1.75*fs

        electrodes = [3,4,5,6,7,8,9,10,11,12]
        powers = np.zeros((len(electrodes),))

        for e in electrodes:
            lfp = simData['LFP'][s:,e]
            freq_wel, ps_wel = ss.welch(lfp,fs=fs,nperseg=nperseg)
            powers[e] = ps_wel[38:42]
        
        fitness = (10**4)*np.mean(powers)

        info = '; '.join(['%s power=%.1f fit=%1.f' % (p, fitness) for p in power])
        print('  ' + info)

        return fitness


    b = Batch(params=params, netParamsFile='netParams_ASSR.py', cfgFile='cfg_ASSR.py', initCfg=initCfg, groupedParams=groupedParams)
    
    b.method = 'optuna'
    
    b.optimCfg = {
        'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
        'fitnessFuncArgs': fitnessFuncArgs,
        'maxFitness': fitnessFuncArgs['maxFitness'],
        'maxiters':     1e6,    #    Maximum number of iterations (1 iteration = 1 function evaluation)
        'maxtime':      None,    #    Maximum time allowed, in seconds
        'maxiter_wait': 60,
        'time_sleep': 150,
        'popsize': 1  # unused - run with mpi 
    }

    return b


# ----------------------------------------------------------------------------------------------
# Run configurations
# ----------------------------------------------------------------------------------------------
def setRunCfg(b, type='mpi_direct'):
    if type=='mpi_direct':
        b.runCfg = {'type': 'mpi_direct',
            'nodes': 1,
            'coresPerNode': 24,
            'script': 'init.py',
            'mpiCommand': 'mpiexec',
            'skip': True}

# ----------------------------------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------------------------------

if __name__ == '__main__':



    b = spont_batch('data/v34_batch25/trial_2142/trial_2142_cfg.json')

    b.batchLabel = 'ASSR_opt'   
    b.saveFolder = 'data/'+b.batchLabel

    setRunCfg(b, 'mpi_direct')
    b.run() # run batch



    
