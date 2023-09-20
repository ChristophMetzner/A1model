import numpy as np
from netpyne import specs
from netpyne.batch import Batch

''' Example of evolutionary algorithm optimization of a cell using NetPyNE

To run use: mpiexec -np [num_cores] nrniv -mpi batch.py
'''

# ---------------------------------------------------------------------------------------------- #
def evolCellITS4():
    # parameters space to explore
    params = specs.ODict()

    scalingRange = [0.25, 4.0]
    scalingRangeReduced = [0.75, 1.5]
    

    params[('tune', 'L')] = scalingRangeReduced
    params[('tune', 'diam')] = scalingRange
    params[('tune', 'Ra')] = scalingRange
    params[('tune', 'cm')] = scalingRange
    params[('tune', 'kv', 'gbar')] = scalingRange
    params[('tune', 'naz', 'gmax')] = scalingRange
    params[('tune', 'pas', 'e')] = scalingRangeReduced
    params[('tune', 'pas', 'g')] = scalingRange

    params[('tune', 'Nca', 'gmax')] = scalingRange
    params[('tune', 'kca', 'gbar')] = scalingRange
    params[('tune', 'km', 'gbar')] = scalingRange


    # current injection params
    interval = 2000
    dur = 500  # ms
    amps = list(np.arange(0.0, 0.65, 0.05))  # amplitudes
    times = list(np.arange(interval, (dur+interval) * len(amps), dur+interval))  # start times
    targetRates = [0., 0., 19., 29., 37., 45., 51., 57., 63., 68., 73., 77., 81.]
 
    stimWeights = [10, 50, 100, 150]
    stimRate = 80
    stimDur = 1000
    stimTimes = [times[-1] + x for x in list(np.arange(interval, (stimDur + interval) * len(stimWeights), stimDur + interval))]
    stimTargetSensitivity = 75
 
    # initial cfg set up
    initCfg = {} # specs.ODict()
    initCfg['duration'] = stimTimes[-1] + stimDur
    initCfg[('hParams', 'celsius')] = 37

    initCfg['savePickle'] = True
    initCfg['saveJson'] = False
    initCfg['saveDataInclude'] = ['simConfig', 'netParams', 'net', 'simData']

    # iclamp
    initCfg[('IClamp1', 'pop')] = 'ITS4'
    initCfg[('IClamp1', 'amp')] = amps
    initCfg[('IClamp1', 'start')] = times
    initCfg[('IClamp1', 'dur')] = dur

    initCfg[('analysis', 'plotTraces', 'timeRange')] = [0, initCfg['duration']] 
    initCfg[('analysis', 'plotfI', 'amps')] = amps
    initCfg[('analysis', 'plotfI', 'times')] = times
    initCfg[('analysis', 'plotfI', 'dur')] = dur
    initCfg[('analysis', 'plotfI', 'targetRates')] = targetRates

    # netstim 
    initCfg[('NetStim1', 'weight')] = stimWeights
    initCfg[('NetStim1', 'start')] = stimTimes
    initCfg[('NetStim1', 'interval')] = 1000.0 / stimRate 
    initCfg[('NetStim1', 'pop')] = 'ITS4'
    initCfg[('NetStim1', 'sec')] = 'soma'
    initCfg[('NetStim1', 'synMech')] = ['AMPA', 'NMDA']
    initCfg[('NetStim1', 'synMechWeightFactor')] = [0.5, 0.5]
    initCfg[('NetStim1', 'number')] = stimRate * stimDur/1000. * 1.1
    initCfg[('NetStim1', 'noise')] = 1.0


    initCfg['removeWeightNorm'] = False
    initCfg[('analysis', 'plotRaster')] = False
    initCfg['printPopAvgRates'] = [[x, x+stimDur] for x in stimTimes]
    
    for k, v in params.items():
        initCfg[k] = v[0]  # initialize params in cfg so they can be modified    

    # fitness function
    fitnessFuncArgs = {}
    fitnessFuncArgs['targetRates'] = targetRates
    
    def fitnessFunc(simData, **kwargs):
        targetRates = kwargs['targetRates']
            
        diffRates = [abs(x - t) for x, t in zip(simData['fI'], targetRates)]
        
        # calculate sensitivity (firing rate) to exc syn inputs 
        stimMaxRate = np.max(list(simData['popRates']['ITS4'].values()))
        
        maxFitness = 1000

        fitness = np.mean(diffRates) if stimMaxRate < stimTargetSensitivity else \
                  np.mean(diffRates) + (stimMaxRate - stimTargetSensitivity)

        
        print(' Candidate rates: ', simData['fI'])
        print(' Target rates:    ', targetRates)
        print(' Difference:      ', diffRates)
        print(' Stim sensitivity: ', stimMaxRate)

        return fitness
        

    # create Batch object with paramaters to modify, and specifying files to use
    b = Batch(params=params, initCfg=initCfg)
    
    b.method = 'optuna'

    if b.method == 'evol':
        # Set output folder, grid method (all param combinations), and run configuration
        b.batchLabel = 'ITS4_evol1'
        b.saveFolder = 'data/'+b.batchLabel
        b.runCfg = {
            'type': 'mpi_bulletin',#'hpc_slurm', 
            'script': 'init.py',
            'mpiCommand': '',
            'nrnCommand': 'python3'
        }

        b.evolCfg = {
            'evolAlgorithm': 'custom',
            'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
            'fitnessFuncArgs': fitnessFuncArgs,
            'pop_size': 95,
            'num_elites': 1, # keep this number of parents for next generation if they are fitter than children
            'mutation_rate': 0.4,
            'crossover': 0.5,
            'maximize': False, # maximize fitness function?
            'max_generations': 2000,
            'time_sleep': 10, # wait this time before checking again if sim is completed (for each generation)
            'maxiter_wait': 6, # max number of times to check if sim is completed (for each generation)
            'defaultFitness': 1000 # set fitness value in case simulation time is over
        }

    elif b.method == 'optuna':
        # Set output folder, grid method (all param combinations), and run configuration
        b.batchLabel = 'ITS4_optuna3'
        b.saveFolder = 'data/'+b.batchLabel
        b.runCfg = {
            'type': 'mpi_direct', #'hpc_slurm', 
            'script': 'init.py',
            'mpiCommand': '',
            'nrnCommand': 'python3'
        }
        b.optimCfg = {
            'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
            'fitnessFuncArgs': fitnessFuncArgs,
            'maxFitness': 1000,
            'maxiters':     2000*95,    #    Maximum number of iterations (1 iteration = 1 function evaluation)
            'maxtime':      100*2000,    #    Maximum time allowed, in seconds
            'maxiter_wait': 6,
            'time_sleep': 10,
            'popsize': 1  # unused - run with mpi 
        }    # Run batch simulations
    
    # Run batch simulations
    b.run()


# ---------------------------------------------------------------------------------------------- #
def evolCellNGF():
    # parameters space to explore
    params = specs.ODict()

    scalingRange = [0.5, 2.0]
    scalingRangeReduced = [0.75, 1.5]
    
    params[('tune', 'L')] = scalingRangeReduced
    params[('tune', 'diam')] = scalingRange
    params[('tune', 'Ra')] = scalingRange
    params[('tune', 'cm')] = scalingRangeReduced
    params[('tune', 'pas', 'e')] = scalingRangeReduced
    params[('tune', 'pas', 'g')] = scalingRange
    params[('tune', 'ch_CavL', 'gmax')] = scalingRange
    params[('tune', 'ch_CavN', 'gmax')] = scalingRange
    params[('tune', 'ch_KCaS', 'gmax')] = scalingRange
    params[('tune', 'ch_Kdrfastngf', 'gmax')] = scalingRange
    params[('tune', 'ch_KvAngf', 'gmax')] = scalingRange
    params[('tune', 'ch_KvCaB', 'gmax')] = scalingRange
    params[('tune', 'ch_Navngf', 'gmax')] = scalingRange
    params[('tune', 'hd', 'gbar')] = scalingRange

    # params[('tune', 'hd', 'ehd')] = scalingRange
    # params[('tune', 'hd', 'elk')] = scalingRange
    # params[('tune', 'hd', 'vhalfl')] = scalingRange
    # params[('tune', 'iconc_Ca', 'caiinf')] = scalingRange
    # params[('tune', 'iconc_Ca', 'catau')] = scalingRange


    # current injection params
    interval = 10000  # 10000
    dur = 500  # ms
    durSteady = 200  # ms
    amps = [0] + list(np.arange(0.04+0.075, 0.121+0.075, 0.01))  # amplitudes
    times = list(np.arange(interval, (dur+interval) * len(amps), dur+interval))  # start times
    targetRatesOnset = [0., 43., 52., 68., 80., 96., 110., 119., 131., 139.]
    targetRatesSteady = [0., 22., 24., 27., 30., 33., 35., 37., 39., 41.]

    stimWeights = [10, 50, 100, 150]
    stimRate = 80
    stimDur = 2000
    stimTimes = [times[-1] + x for x in list(np.arange(interval, (stimDur + interval) * len(stimWeights), stimDur + interval))]
    stimTargetSensitivity = 200 

    # initial cfg set up
    initCfg = {} # specs.ODict()
    initCfg['duration'] = stimTimes[-1] + stimDur # ((dur+interval) * len(amps)) + ((stimDur+interval) * len(stimWeights)) 
    initCfg[('hParams', 'celsius')] = 37

    initCfg['savePickle'] = True
    initCfg['saveJson'] = False
    initCfg['saveDataInclude'] = ['simConfig', 'netParams', 'net', 'simData']

    initCfg[('IClamp1', 'pop')] = 'NGF'
    initCfg[('IClamp1', 'amp')] = amps
    initCfg[('IClamp1', 'start')] = times
    initCfg[('IClamp1', 'dur')] = dur
    
    # iclamp
    initCfg[('analysis', 'plotTraces', 'timeRange')] = [0, initCfg['duration']] 
    initCfg[('analysis', 'plotfI', 'amps')] = amps
    initCfg[('analysis', 'plotfI', 'times')] = times
    initCfg[('analysis', 'plotfI', 'calculateOnset')] = True
    initCfg[('analysis', 'plotfI', 'dur')] = dur
    initCfg[('analysis', 'plotfI', 'durSteady')] = durSteady
    initCfg[('analysis', 'plotfI', 'targetRates')] = [] #
    initCfg[('analysis', 'plotfI', 'targetRatesOnset')] = targetRatesOnset
    initCfg[('analysis', 'plotfI', 'targetRatesSteady')] = targetRatesSteady

    # netstim 
    initCfg[('NetStim1', 'weight')] = stimWeights
    initCfg[('NetStim1', 'start')] = stimTimes
    initCfg[('NetStim1', 'interval')] = 1000.0 / stimRate 
    initCfg[('NetStim1', 'pop')] = 'NGF'
    initCfg[('NetStim1', 'sec')] = 'soma'
    initCfg[('NetStim1', 'synMech')] = ['AMPA', 'NMDA']
    initCfg[('NetStim1', 'synMechWeightFactor')] = [0.5, 0.5]
    initCfg[('NetStim1', 'number')] = stimRate * stimDur/1000. * 1.1
    initCfg[('NetStim1', 'noise')] = 1.0


    initCfg['removeWeightNorm'] = False
    initCfg[('analysis', 'plotRaster')] = False
    initCfg['printPopAvgRates'] = [[x, x+stimDur] for x in stimTimes]
    
    
    for k, v in params.items():
        initCfg[k] = v[0]  # initialize params in cfg so they can be modified    

    # fitness function
    fitnessFuncArgs = {}
    fitnessFuncArgs['targetRatesOnset'] = targetRatesOnset
    fitnessFuncArgs['targetRatesSteady'] = targetRatesSteady
    fitnessFuncArgs['stimTargetSensitivity'] = stimTargetSensitivity
    
    def fitnessFunc(simData, **kwargs):
        targetRatesOnset = kwargs['targetRatesOnset']
        targetRatesSteady = kwargs['targetRatesSteady']
        stimTargetSensitivity = kwargs['stimTargetSensitivity']
            
        # fI curve
        diffRatesOnset = [abs(x-t) for x,t in zip(simData['fI_onset'], targetRatesOnset)]
        diffRatesSteady = [abs(x - t) for x, t in zip(simData['fI_steady'], targetRatesSteady)]
        
        # for spontaneous (fI with 0 nA current) use regular fI calculation to avoid missing spikes  
        # and penalize x10 to avoid these solutions
        diffRatesOnset[0] = abs(simData['fI'][0] - targetRatesOnset[0]) * 10
        diffRatesSteady[0] = abs(simData['fI'][0] - targetRatesSteady[0]) * 10

        # calculate sensitivity (firing rate) to exc syn inputs 
        stimMaxRate = np.max(list(simData['popRates']['NGF'].values()))
        # stimMinRate = np.min(list(simData['popRates']['NGF'].values()))
        # stimDiffRate = stimMaxRate - stimMinRate
        
        maxFitness = 1000
        #fitness = np.mean(diffRatesOnset + diffRatesSteady) + np.max([0, stimMaxRate - stimTargetSensitivity])
        fitness = np.mean(diffRatesOnset + diffRatesSteady) if stimMaxRate < stimTargetSensitivity else maxFitness

        print(' Candidate rates: ', simData['fI_onset']+simData['fI_steady'])
        print(' Target rates:    ', targetRatesOnset+targetRatesSteady)
        print(' Difference:      ', diffRatesOnset + diffRatesSteady)
        print(' Stim sensitivity: ', stimMaxRate)

        return fitness
        

    # create Batch object with paramaters to modify, and specifying files to use
    b = Batch(params=params, initCfg=initCfg) 
    b.method = 'optuna'

    if b.method == 'evol':
        # Set output folder, grid method (all param combinations), and run configuration
        b.batchLabel = 'NGF_evol3'
        b.saveFolder = 'data/'+b.batchLabel
        b.runCfg = {
            'type': 'mpi_bulletin',#'hpc_slurm', 
            'script': 'init.py',
            'mpiCommand': '',
            'nrnCommand': 'python3'
        }

        b.evolCfg = {
            'evolAlgorithm': 'custom',
            'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
            'fitnessFuncArgs': fitnessFuncArgs,
            'pop_size': 95,
            'num_elites': 1, # keep this number of parents for next generation if they are fitter than children
            'mutation_rate': 0.4,
            'crossover': 0.5,
            'maximize': False, # maximize fitness function?
            'max_generations': 2000,
            'time_sleep': 10, # wait this time before checking again if sim is completed (for each generation)
            'maxiter_wait': 6, # max number of times to check if sim is completed (for each generation)
            'defaultFitness': 1000 # set fitness value in case simulation time is over
        }

    elif b.method == 'optuna':
        # Set output folder, grid method (all param combinations), and run configuration
        b.batchLabel = 'NGF_optuna3'
        b.saveFolder = 'data/'+b.batchLabel
        b.runCfg = {
            'type': 'mpi_direct', #'hpc_slurm', 
            'script': 'init.py',
            'mpiCommand': '',
            'nrnCommand': 'python3'
        }
        b.optimCfg = {
            'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
            'fitnessFuncArgs': fitnessFuncArgs,
            'maxFitness': 1000,
            'maxiters':     2000*95,    #    Maximum number of iterations (1 iteration = 1 function evaluation)
            'maxtime':      100*2000,    #    Maximum time allowed, in seconds
            'maxiter_wait': 6,
            'time_sleep': 10,
            'popsize': 1  # unused - run with mpi 
        }


    # Run batch simulations
    b.run()


# ---------------------------------------------------------------------------------------------- #
# Main code
if __name__ == '__main__':
    evolCellITS4()
    #evolCellNGF() 

