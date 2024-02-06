import subprocess as sp

seeds = [12345]#, 23451]#, 34512, 45123, 51234]

for seed in seeds:
	label = 'test'+str(seed) 
	call  = 'mpiexec -n 12 nrniv -python -mpi init_manual_batch.py '+str(seed)+' '+label
	print(call)
	sp.call([call],shell=True)
