import numpy as np
from pylab import *
import pickle
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import OrderedDict
from utils import getdatestr
from scipy.stats import pearsonr

rcParams['agg.path.chunksize'] = 100000000000 # for plots of long activity 
ion()

rcParams['font.size'] = 12
tl=tight_layout

def loadsimdat (name,lpop = []): # load simulation data
  global totalDur, tstepPerAction
  print('loading data from', name)
  simConfig = pickle.load(open('data/'+name+'/'+name+'_data.pkl','rb'))
  dstartidx,dendidx={},{} # starting,ending indices for each population
  for p in simConfig['net']['pops'].keys():
    if simConfig['net']['pops'][p]['numCells'] > 0:
      dstartidx[p] = simConfig['net']['pops'][p]['cellGids'][0]
      dendidx[p] = simConfig['net']['pops'][p]['cellGids'][-1]
  dnumc = {}
  for p in simConfig['net']['pops'].keys():
    if p in dstartidx:
      dnumc[p] = dendidx[p]-dstartidx[p]+1
    else:
      dnumc[p] = 0
  spkID= np.array(simConfig['simData']['spkid'])
  spkT = np.array(simConfig['simData']['spkt'])
  dspkID,dspkT = {},{}
  for pop in simConfig['net']['pops'].keys():
    if dnumc[pop] > 0:
      dspkID[pop] = spkID[(spkID >= dstartidx[pop]) & (spkID <= dendidx[pop])]
      dspkT[pop] = spkT[(spkID >= dstartidx[pop]) & (spkID <= dendidx[pop])]
  totalDur = simConfig['simData']['t'][-1] 
  # tstepPerAction = dconf['sim']['tstepPerAction'] # time step per action (in ms)  
  return simConfig, dstartidx, dendidx, dnumc, dspkID, dspkT

def getspikehist (spkT, numc, binsz, tmax):
  tt = np.arange(0,tmax,binsz)
  nspk = [len(spkT[(spkT>=tstart) & (spkT<tstart+binsz)]) for tstart in tt]
  nspk = [1e3*x/(binsz*numc) for x in nspk]
  return tt,nspk

def getspikehistpops (spkT, lk, dnumc, binsz, tmax):
  # get binned firing rate time-series from multiple populations
  tt = np.arange(0,tmax,binsz)
  nspk = np.zeros((len(tt),))
  for k in lk:  nspk = nspk + np.array([len(spkT[k][(spkT[k]>=tstart) & (spkT[k]<tstart+binsz)]) for tstart in tt])
  numc = np.sum([dnumc[k] for k in lk])  
  nspk = [1e3*x/(binsz*numc) for x in nspk]
  return tt,nspk

#
def getrate (dspkT,dspkID, pop, dnumc, tlim=None):
  # get average firing rate for the population, over entire simulation
  nspk = len(dspkT[pop])
  ncell = dnumc[pop]
  if tlim is not None:
    spkT = dspkT[pop]
    nspk = len(spkT[(spkT>=tlim[0])&(spkT<=tlim[1])])
    return 1e3*nspk/((tlim[1]-tlim[0])*ncell)
  else:  
    return 1e3*nspk/(totalDur*ncell)

def pravgrates (dspkT,dspkID,dnumc,tlim=None):
  # print average firing rates over simulation duration
  for pop in dspkT.keys(): print(pop,round(getrate(dspkT,dspkID,pop,dnumc,tlim=tlim),2),'Hz')

#
def drawraster (dspkT,dspkID,tlim=None,msz=2,skipstim=True,cmap=None):
  # draw raster (x-axis: time, y-axis: neuron ID)
  lpop=list(dspkT.keys()); lpop.reverse()
  lpop = [x for x in lpop if not skipstim or x.count('stim')==0]  
  csm=cm.ScalarMappable(cmap=cmap); csm.set_clim(0,len(dspkT.keys())) # cm.tab20 is a decent colormap
  lclr = []
  for pdx,pop in enumerate(lpop):
    if cmap is None:
      if pop.count('PV')>0 or pop.count('NGF')>0 or pop.count('VIP')>0 or pop.count('SOM')>0:
        color = 'b'
      elif pop.count('TIM')>0 or pop.count('TI')>0 or pop.count('IREM')>0 or pop.count('IRE')>0:
        color = 'g'
      else:
        color = 'r'
    else:
      color = csm.to_rgba(pdx);
    lclr.append(color)
    plot(dspkT[pop],dspkID[pop],'o',color=color,markersize=msz)
  if tlim is not None:
    xlim(tlim)
  else:
    xlim((0,totalDur))
  xlabel('Time (ms)')
  #lclr.reverse(); 
  lpatch = [mpatches.Patch(color=c,label=s+' '+str(round(getrate(dspkT,dspkID,s,dnumc),2))+' Hz') for c,s in zip(lclr,lpop)]
  lpatch.reverse()
  ax=gca()
  ax.legend(handles=lpatch,handlelength=1,loc='best')
  ylim((0,sum([dnumc[x] for x in lpop])))
  ax.invert_yaxis() # so superficial layers at top of raster plot

#
def drawcellVm (simConfig, ldrawpop=None,tlim=None, lclr=None,cmap=cm.prism):
  csm=cm.ScalarMappable(cmap=cmap); csm.set_clim(0,len(dspkT.keys()))
  if tlim is not None:
    dt = simConfig['simData']['t'][1]-simConfig['simData']['t'][0]    
    sidx,eidx = int(0.5+tlim[0]/dt),int(0.5+tlim[1]/dt)
  dclr = OrderedDict(); lpop = []
  for kdx,k in enumerate(list(simConfig['simData']['V_soma'].keys())):  
    color = csm.to_rgba(kdx);
    if lclr is not None and kdx < len(lclr): color = lclr[kdx]
    cty = simConfig['net']['cells'][int(k.split('_')[1])]['cellType']
    if ldrawpop is not None and cty not in ldrawpop: continue
    dclr[kdx]=color
    lpop.append(simConfig['net']['cells'][int(k.split('_')[1])]['cellType'])
  if ldrawpop is None: ldrawpop = lpop    
  for kdx,k in enumerate(list(simConfig['simData']['V_soma'].keys())):
    cty = simConfig['net']['cells'][int(k.split('_')[1])]['cellType']
    if ldrawpop is not None and cty not in ldrawpop: continue
    if tlim is not None:
      plot(simConfig['simData']['t'][sidx:eidx],simConfig['simData']['V_soma'][k][sidx:eidx],color=dclr[kdx])
    else:
      plot(simConfig['simData']['t'],simConfig['simData']['V_soma'][k],color=dclr[kdx])      
  lpatch = [mpatches.Patch(color=c,label=s) for c,s in zip(dclr.values(),ldrawpop)]
  ax=gca()
  ax.legend(handles=lpatch,handlelength=1,loc='best')
  if tlim is not None: ax.set_xlim(tlim)

def gifpath (): global simstr; return 'gif/' + getdatestr() + simstr

if __name__ == '__main__':
  global simstr
  name = '23aug24_BBN0'  
  if len(sys.argv) > 0: name = sys.argv[1]
  simConfig, dstartidx, dendidx, dnumc, dspkID, dspkT = loadsimdat(name,lpop=[])
  dstr = getdatestr(); simstr = name # date and sim string
  print('loaded simulation data',simstr,'on',dstr)
  if totalDur <= 10e3:
    pravgrates(dspkT,dspkID,dnumc,tlim=(totalDur-1e3,totalDur))
    drawraster(dspkT,dspkID)
    # figure(); drawcellVm(simConfig,lclr=['r','g','b','c','m','y'])    
  else:
    pravgrates(dspkT,dspkID,dnumc,tlim=(250,totalDur))

  
