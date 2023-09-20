"""
init.py

Starting script to run NetPyNE-based A1 model.


Usage:
    python init.py # Run simulation, optionally plot a raster


MPI usage:
    mpiexec -n 4 nrniv -python -mpi init.py


Contributors: ericaygriffith@gmail.com, salvadordura@gmail.com
"""

import matplotlib; matplotlib.use('Agg')  # to avoid graphics error in servers

from netpyne import sim

cfg, netParams = sim.readCmdLineArgs(simConfigDefault='cfg_cell.py', netParamsDefault='netParams_cell.py')
#sim.create(netParams, cfg)
#sim.gatherData()

sim.create(netParams, cfg)
sim.simulate()
sim.analyze()
