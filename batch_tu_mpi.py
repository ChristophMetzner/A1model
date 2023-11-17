"""
batch.py 

Batch simulation for M1 model using NetPyNE

Contributors: salvadordura@gmail.com
"""
from netpyne.batch import Batch
from netpyne import specs
import numpy as np


# ----------------------------------------------------------------------------------------------
# Spontaneous activity different seeds
# ----------------------------------------------------------------------------------------------
def spont_batch(filename):
    params = specs.ODict()

    if not filename:
        filename = 'data/v34_batch25/trial_2142/trial_2142_cfg.json'

    # from prev 
    import json
    with open(filename, 'rb') as f:
        cfgLoad = json.load(f)['simConfig']
    cfgLoad2 = cfgLoad

    # #### SET CONN AND STIM SEEDS #### 
    params[('seeds', 'conn')] = [12345, 23451, 34512, 45123, 51234, 67890, 6789, 90678, 89067, 78906]

    #### GROUPED PARAMS #### 
    groupedParams = [] 

    # --------------------------------------------------------
    # initial config
    initCfg = {} # set default options from prev sim
    
    initCfg['duration'] = 4000 #11500 
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



    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg_spont.py', initCfg=initCfg, groupedParams=groupedParams)
    #b = Batch(params=params, netParamsFile='netParams_SST.py', cfgFile='cfg_spont.py', initCfg=initCfg, groupedParams=groupedParams)
    #b = Batch(params=params, netParamsFile='netParams_PV.py', cfgFile='cfg_spont.py', initCfg=initCfg, groupedParams=groupedParams)
    #b = Batch(params=params, netParamsFile='netParams_NMDA.py', cfgFile='cfg_spont.py', initCfg=initCfg, groupedParams=groupedParams)
    b.method = 'grid'

    return b


# ----------------------------------------------------------------------------------------------
# Run configurations
# ----------------------------------------------------------------------------------------------
def setRunCfg(b, type='mpi_direct'):
    if type=='mpi_direct':
        b.runCfg = {'type': 'mpi_direct',
            'cores':2,
            'script': 'init.py',
            'mpiCommand': 'mpiexec',
            'skip': True}

# ----------------------------------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------------------------------

if __name__ == '__main__':



    b = spont_batch('data/v34_batch25/trial_2142/trial_2142_cfg.json')

    b.batchLabel = 'test'   
    b.saveFolder = 'data/'+b.batchLabel

    setRunCfg(b, 'mpi_direct')
    b.run() # run batch



    
