### https://github.com/suny-downstate-medical-center/S1_netpyne/blob/samn/sim/simdat.py
### FROM FERNANDO; CONVERT DIPOLES TO MATLAB FORMAT FROM NETPYNE

import pickle
from pylab import *
import matplotlib.patches as mpatches
import scipy
tl=tight_layout
ion()
totalDur = 0

def loaddat (simname):
  global totalDur
  d = pickle.load(open(simname,'rb'))   # '../data/'+simname+'/'+simname+'_data.pkl'
  simConfig = d # ['simConfig']
  sdat = d['simData']
  totalDur = d['simConfig']['duration']
  dstartidx,dendidx={},{} # starting,ending indices for each population
  for p in simConfig['net']['params']['popParams'].keys():
    if 'tags' in simConfig['net']['pops'][p]:
      numCells = simConfig['net']['pops'][p]['tags']['numCells']
    else:
      numCells = simConfig['net']['pops'][p]['numCells']
    if numCells > 0:
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
  return simConfig, sdat, dstartidx, dendidx, dnumc, dspkID, dspkT

def getspikehist (spkT, numc, binsz, tmax):
  tt = np.arange(0,tmax,binsz)
  nspk = [len(spkT[(spkT>=tstart) & (spkT<tstart+binsz)]) for tstart in tt]
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
def drawraster (dspkT,dspkID,dnumc,tlim=None,msz=2,skipstim=True,drawlegend=False,saveFig=True,rasterFile=None):
  # draw raster (x-axis: time, y-axis: neuron ID)
  # NOTE: ADDING dnumc TO THE ARGS FOR IF drawLegend IS TRUE! 
  lpop=list(dspkT.keys())#; lpop.reverse()
  # print('lpop: ' + str(lpop))
  lpop = [x for x in lpop if not skipstim or x.count('stim')==0]  
  
  ## EYG ADDING COLORLIST 9/23/22
  colorList = [[0.42, 0.67, 0.84], [0.90, 0.76, 0.00], [0.42, 0.83, 0.59], [0.90, 0.32, 0.00], [0.34, 0.67, 0.67], [0.90, 0.59, 0.00], [0.42, 0.82, 0.83], [1.00, 0.85, 0.00], [0.33, 0.67, 0.47], [1.00, 0.38, 0.60], [0.57, 0.67, 0.33], [0.50, 0.20, 0.00], [0.71, 0.82, 0.41], [0.00, 0.20, 0.50], [0.70, 0.32, 0.10]] * 4  # *3
  
  popColors = {pop: colorList[pdx % len(colorList)] for pdx, pop in enumerate(lpop)}
  # print('popColors: ' + str(popColors))

  lclr = []
  figure(figsize=(9*0.95, 13*0.9))
  for pdx,pop in enumerate(lpop):
    # popColors = {pop: colorList[pdx % len(colorList)]}
    print('plotting pop: ' + str(pop))
    lclr.append(popColors[pop])
    # plot(dspkT[pop],dspkID[pop],'.',color=popColors[pop],markersize=msz)  # '.' instead of 'o'
    ### EYG 9/23/22 -- TRY SCATTER INSTEAD OF PLOT
    scatter(dspkT[pop],dspkID[pop],s=1,color=popColors[pop],marker='.') #,markersize=msz)  # '.' instead of 'o'
    ### ^^ CINECA WANTS 'color' keyword INSTEAD OF 'c' it would seem!!)

  
  ###   
  
  # ## ORIG COLOR LINES + RASTER PLOTTING  ## COMMENTING OUT, EYG 9/23/22
  # csm=cm.ScalarMappable(cmap=cm.prism); csm.set_clim(0,len(dspkT.keys()))
  # lclr = []
  # for pdx,pop in enumerate(lpop):
  #   color = csm.to_rgba(pdx); lclr.append(color)
  #   plot(dspkT[pop],dspkID[pop],'.',color=color,markersize=msz)  # '.' instead of 'o'
  # ### 

  if tlim is not None:
    xlim(tlim)
  else:
    xlim((0,totalDur))
  xlabel('Time (ms)')
  #lclr.reverse();
  if drawlegend:
    # EYG 9/23/22
    ####### ORIG: #######
    ## lpatch = [mpatches.Patch(color=c,label=s+' '+str(round(getrate(dspkT,dspkID,s,dnumc,tlim=tlim),2))+' Hz') for c,s in zip(lclr,lpop)]
    ## ax=gca()
    ## ax.legend(handles=lpatch,handlelength=0.5,loc=2,fontsize=5,bbox_to_anchor=(1.025, 1),borderaxespad=0.0)     ## ADDING ARGS TO legend(); EYG 9/23/22 -->  handlelength=0.5 (from 1.0), loc=2 (from 'best'), fontsize='small' (from 5),bbox_to_anchor, borderaxespad

    ####### NEW LEGEND CODE #######
    leg_labels = []
    handles = []
    for pdx,pop in enumerate(lpop):
      leg_labels.append(pop)
      handles.append(mpatches.Rectangle((0, 0), 1, 1, fc=popColors[pop]))
    ax=gca()
    ax.legend(handles,leg_labels, loc=2, borderaxespad=0.0, handlelength=0.5, fontsize=5, bbox_to_anchor=(1.025, 1)) 
    # tight_layout() -- putting this later before save!! EYG 9/23/22
    ########### 

  # EYG 9/23/22
  ## ORIG: ##
  # ylim((0,sum([dnumc[x] for x in lpop])))
  ## NEW: ##
  ylim0, ylim1 = ax.get_ylim()
  print('ylim1: ' + str(ylim1))
  print('ylim0: ' + str(ylim0))
  # ax.set_ylim(ylim1, ylim0)
  ylim((0,sum([dnumc[x] for x in lpop])))
  print('sum([dnumc[x] for x in lpop]): ' + str(sum([dnumc[x] for x in lpop])))

  # EYG 9/23/22
  ylabel('NEURON ID')

  # EYG 9/23/22
  ax.invert_yaxis()

  # EYG 9/23/22
  tight_layout()
  
  ## EYG ADDING SAVEFIG LINES 9/23/22
  if saveFig:
    if rasterFile is None:
      rasterFile = 'raster.png'
    savefig(rasterFile, dpi=300)
    print('Saved ' + rasterFile)
  # ADD ARG + LINE(S) for showFig??

#
def drawcellVm (simConfig, ldrawpop=None,tlim=None, lclr=None):
  csm=cm.ScalarMappable(cmap=cm.prism); csm.set_clim(0,len(dspkT.keys()))
  if tlim is not None:
    dt = simConfig['simData']['t'][1]-simConfig['simData']['t'][0]    
    sidx,eidx = int(0.5+tlim[0]/dt),int(0.5+tlim[1]/dt)
  dclr = OrderedDict(); lpop = []
  for kdx,k in enumerate(list(simConfig['simData']['V_soma'].keys())):  
    color = csm.to_rgba(kdx);
    if lclr is not None and kdx < len(lclr): color = lclr[kdx]
    cty = simConfig['net']['cells'][int(k.split('_')[1])]['tags']['cellType']
    if ldrawpop is not None and cty not in ldrawpop: continue
    dclr[kdx]=color
    lpop.append(simConfig['net']['cells'][int(k.split('_')[1])]['tags']['cellType'])
  if ldrawpop is None: ldrawpop = lpop    
  for kdx,k in enumerate(list(simConfig['simData']['V_soma'].keys())):
    cty = simConfig['net']['cells'][int(k.split('_')[1])]['tags']['cellType']
    if ldrawpop is not None and cty not in ldrawpop: continue
    if tlim is not None:
      plot(simConfig['simData']['t'][sidx:eidx],simConfig['simData']['V_soma'][k][sidx:eidx],color=dclr[kdx])
    else:
      plot(simConfig['simData']['t'],simConfig['simData']['V_soma'][k],color=dclr[kdx])      
  lpatch = [mpatches.Patch(color=c,label=s) for c,s in zip(dclr.values(),ldrawpop)]
  ax=gca()
  ax.legend(handles=lpatch,handlelength=1,loc='best')
  if tlim is not None: ax.set_xlim(tlim)

def IsCortical (ty): return ty.startswith('L') and ty.count('_') > 0

def IsThal (ty): return not IsCortical(ty)

def GetCellType (idx, dnumc, dstartidx, dendidx):
  for ty in dnumc.keys():
    if idx >= dstartidx[ty] and idx <= dendidx[ty]: return ty
  return -1

def GetCellCoords (simConfig, idx):
  if 'tags' in simConfig['net']['cells'][idx]:
    return [simConfig['net']['cells'][idx]['tags'][k] for k in ['x','y','z']]
  else:
    return [simConfig['net']['cells'][idx][k] for k in ['x','y','z']]    

def save_dipoles_matlab (outfn, simConfig, sdat, dnumc, dstartidx, dendidx):
  # save dipoles in matlab format to file outfn
  # adapting (not done yet) from https://github.com/NathanKlineInstitute/A1/blob/salva_layers/analysis/disc_grant.py#L368
  from scipy import io
  lidx = list(sdat['dipoleCells'].keys())
  lty = [GetCellType(idx,dnumc,dstartidx,dendidx) for idx in lidx]
  # lcort = [IsCortical(ty) for ty in lty]    
  cellPos = [GetCellCoords(simConfig,idx) for idx in lidx]
  cellDipoles = [sdat['dipoleCells'][idx] for idx in lidx]
  #matDat = {'cellPos': cellPos, 'cellPops': lty, 'cellDipoles': cellDipoles, 'dipoleSum': sdat['dipoleSum'], 'cortical': lcort}
  matDat = {'cellPos': cellPos, 'cellPops': lty, 'cellDipoles': cellDipoles, 'dipoleSum': sdat['dipoleSum']}
  io.savemat(outfn, matDat, do_compression=True)