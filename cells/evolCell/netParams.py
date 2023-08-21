
"""
netParams.py 

High-level specifications for M1 network model using NetPyNE

Contributors: salvadordura@gmail.com
"""

from netpyne import specs

try:
    from __main__ import cfg  # import SimConfig object with params from parent module
except:
    from cfg import cfg  # if no simConfig in parent module, import directly from tut8_cfg module

#------------------------------------------------------------------------------
#
# NETWORK PARAMETERS
#
#------------------------------------------------------------------------------

netParams = specs.NetParams()   # object of class NetParams to store the network parameters

#------------------------------------------------------------------------------
# Cell parameters
#------------------------------------------------------------------------------

# cell params
# netParams.loadCellParamsRule('NGF_reduced', 'NGF_reduced_cellParams.json')
# netParams.cellParams['NGF_reduced']['conds'] = {'cellType': 'NGF'}

netParams.loadCellParamsRule('ITS4_reduced', 'ITS4_reduced_cellParams.json')
netParams.cellParams['ITS4_reduced']['conds'] = {'cellType': 'ITS4'}


for sec, secDict in netParams.cellParams['ITS4_reduced']['secs'].items():
    #if sec in cfg.tune:
    # vinit
    if 'vinit' in cfg.tune:
        secDict['vinit'] = cfg.tune['vinit']
    
    # mechs
    for mech in secDict['mechs']:
        if mech in cfg.tune:
            for param in secDict['mechs'][mech]:
                if param in cfg.tune[mech]:
                    secDict['mechs'][mech][param] *= cfg.tune[mech][param]  
    
    # geom
    for geomParam in secDict['geom']:
        if geomParam in cfg.tune:
            secDict['geom'][geomParam] *= cfg.tune[geomParam]


#------------------------------------------------------------------------------
# Population parameters
#------------------------------------------------------------------------------
#netParams.popParams['NGF'] = {'cellType': 'NGF', 'numCells': 1}
netParams.popParams['ITS4'] = {'cellType': 'ITS4', 'numCells': 1}


#------------------------------------------------------------------------------
# Synaptic mechanism parameters
#------------------------------------------------------------------------------
netParams.synMechParams['NMDA'] = {'mod': 'MyExp2SynNMDABB', 'tau1NMDA': 15, 'tau2NMDA': 150, 'e': 0}
netParams.synMechParams['AMPA'] = {'mod':'MyExp2SynBB', 'tau1': 0.05, 'tau2': 5.3, 'e': 0}


#------------------------------------------------------------------------------
# Current inputs (IClamp)
#------------------------------------------------------------------------------
if cfg.addIClamp:	
     for iclabel in [k for k in dir(cfg) if k.startswith('IClamp')]:
        ic = getattr(cfg, iclabel, None)  # get dict with params

        amps = ic['amp'] if isinstance(ic['amp'], list) else [ic['amp']]  # make amps a list if not already
        starts = ic['start'] if isinstance(ic['start'], list) else [ic['start']]  # make amps a list if not already

        for amp, start in zip(amps, starts):    
            # add stim source
            netParams.stimSourceParams[iclabel+'_'+str(amp)] = {'type': 'IClamp', 'delay': start, 'dur': ic['dur'], 'amp': amp}
            
            # connect stim source to target
            netParams.stimTargetParams[iclabel+'_'+ic['pop']+'_'+str(amp)] = \
                {'source': iclabel+'_'+str(amp), 'conds': {'pop': ic['pop']}, 'sec': ic['sec'], 'loc': ic['loc']}

#------------------------------------------------------------------------------
# NetStim inputs (to simulate short external stimuli; not bkg)
#------------------------------------------------------------------------------
if cfg.addNetStim:
    for key in [k for k in dir(cfg) if k.startswith('NetStim')]:
        params = getattr(cfg, key, None)
        [pop, ynorm, sec, loc, synMech, synMechWeightFactor, starts, interval, noise, number, weights, delay] = \
        [params[s] for s in ['pop', 'ynorm', 'sec', 'loc', 'synMech', 'synMechWeightFactor', 'start', 'interval', 'noise', 'number', 'weight', 'delay']] 

        for weight, start in zip(weights, starts):
            # add stim source
            netParams.stimSourceParams[key+'_'+str(start)] = {'type': 'NetStim', 'start': start, 'interval': interval, 'noise': noise, 'number': number}

            # connect stim source to target 
            netParams.stimTargetParams[key+'_'+str(start)+'_'+pop] =  {
                'source': key+'_'+str(start), 
                'conds': {'pop': pop, 'ynorm': ynorm},
                'sec': sec, 
                'loc': loc,
                'synMech': synMech,
                'weight': weight,
                'synMechWeightFactor': synMechWeightFactor,
                'delay': delay}