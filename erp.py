from pylab import *
import numpy as np
import scipy.signal as sps
import os
import pandas as pd

def index2ms (idx, sampr): return 1e3*idx/sampr
def ms2index (ms, sampr): return int(sampr*ms/1e3)

# get the average ERP (dat should be either LFP or CSD)
def getERPOnChan (dat, sampr, chan, trigtimes, swindowms, ewindowms):
  nrow = dat.shape[0]
  tt = np.linspace(swindowms, ewindowms,ms2index(ewindowms - swindowms,sampr))
  swindowidx = ms2index(swindowms,sampr) # could be negative
  ewindowidx = ms2index(ewindowms,sampr)
  lERP = np.zeros((len(trigtimes),len(tt)))
  for i,trigidx in enumerate(trigtimes): # go through stimuli
    sidx = max(0,trigidx+swindowidx)
    eidx = min(dat.shape[1],trigidx+ewindowidx)
    lERP[i,:] = dat[chan, sidx:eidx]
  return tt,lERP

# get the average ERP (dat should be either LFP or CSD)
def getAvgERP (dat, sampr, trigtimes, swindowms, ewindowms):
  nrow = dat.shape[0]
  tt = np.linspace(swindowms, ewindowms,ms2index(ewindowms - swindowms,sampr))
  swindowidx = ms2index(swindowms,sampr) # could be negative
  ewindowidx = ms2index(ewindowms,sampr)
  avgERP = np.zeros((nrow,len(tt)))
  for chan in range(nrow): # go through channels
    for trigidx in trigtimes: # go through stimuli
      sidx = max(0,trigidx+swindowidx)
      eidx = min(dat.shape[1],trigidx+ewindowidx)
      avgERP[chan,:] += dat[chan, sidx:eidx]
    avgERP[chan,:] /= float(len(trigtimes))
  return tt,avgERP

# draw the average ERP (dat should be either LFP or CSD)
def drawAvgERP (dat, sampr, trigtimes, swindowms, ewindowms, whichchan=None, yl=None, clr=None,lw=1):
  ttavg,avgERP = getAvgERP(dat,sampr,trigtimes,swindowms,ewindowms)
  nrow = avgERP.shape[0]
  for chan in range(nrow): # go through channels
    if whichchan is None:
      subplot(nrow,1,chan+1)
      plot(ttavg,avgERP[chan,:],color=clr,linewidth=lw)
    elif chan==whichchan:
      plot(ttavg,avgERP[chan,:],color=clr,linewidth=lw)
    xlim((-swindowms,ewindowms))
    if yl is not None: ylim(yl)
  
# draw the event related potential (or associated CSD signal), centered around stimulus start (aligned to t=0)
def drawERP (dat, sampr, trigtimes, windowms, whichchan=None, yl=None,clr=None,lw=1):
  if clr is None: clr = 'gray'
  nrow = dat.shape[0]
  tt = np.linspace(-windowms,windowms,ms2index(windowms*2,sampr))
  windowidx = ms2index(windowms,sampr)
  for trigidx in trigtimes: # go through stimuli
    for chan in range(nrow): # go through channels
      sidx = max(0,trigidx-windowidx)
      eidx = min(dat.shape[1],trigidx+windowidx)
      if whichchan is None:
        subplot(nrow,1,chan+1)
        plot(tt,dat[chan, sidx:eidx],color=clr,linewidth=lw)
      elif chan==whichchan:
        plot(tt,dat[chan, sidx:eidx],color=clr,linewidth=lw)
      xlim((-windowms,windowms))
      if yl is not None: ylim(yl)
      #xlabel('Time (ms)')

# normalized cross-correlation between x and y
def normcorr (x, y):
  # Pad shorter array if signals are different lengths
  if x.size > y.size:
    pad_amount = x.size - y.size
    y = np.append(y, np.repeat(0, pad_amount))
  elif y.size > x.size:
    pad_amount = y.size - x.size
    x = np.append(x, np.repeat(0, pad_amount))
  corr = np.correlate(x, y, mode='full')  # scale = 'none'
  lags = np.arange(-(x.size - 1), x.size)
  corr /= np.sqrt(np.dot(x, x) * np.dot(y, y))
  return lags, corr

# x is longer signal; y is short pattern; nsamp is moving window size (in samples) for finding pattern
def windowcorr (x, y, nsamp, verbose=False):
  sz = len(x)
  lsidx,leidx=[],[]
  llag, lc = [],[]
  for sidx in range(0,sz,nsamp):
    lsidx.append(sidx)
    eidx = min(sidx + nsamp, sz-1)
    leidx.append(eidx)
    if verbose: print(sidx,eidx)
    sig = sps.detrend(x[sidx:eidx])
    lags,c = normcorr(sig,y)
    llag.append(lags[int(len(lags)/2):])
    lc.append(c[int(len(lags)/2):])
  return llag, lc, lsidx, leidx

#
def maxnormcorr (x, y):
  lags, corr = normcorr(x,y)
  return max(corr)

#
def maxnormcorrlag (x, y):
  lags, corr = normcorr(x,y)
  return lags[np.argmax(corr)]

