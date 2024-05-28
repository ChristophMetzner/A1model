#
# Generate cochlear input spikes to use auditory thalamocortical model
import numpy as np
import soundfile as sf # pip install soundfile

rng = np.random.RandomState()

def readwav (fn):
  # read a wav file 
  dat, sampr = sf.read(fn)
  return dat, sampr

def writewav (dat, fn, sampr):
  # write array dat to wav file fn using sampling rate sampr
  sf.write(fn, dat, sampr)

def cochlearSpikes (freqRange=[125, 20000],numCenterFreqs=100,loudnessScale=1, lfnwave=[],lonset=[]):
  dout = {}; i = 0
  for fnwave,onset in zip(lfnwave,lonset):
    d = cochlearInputSpikes(freqRange=freqRange,numCenterFreqs=numCenterFreqs,loudnessScale=loudnessScale,fnwave=fnwave,onset=onset)
    if i == 0:
      dout = d
    else:
      for spktA,spktB in zip(d['spkT'],dout['spkT']):
        for t in spktA: spktB.append(t)
    i += 1
  return dout

def cochlearInputSpikes (freqRange=[125, 20000],
                         numCenterFreqs=100,
                         loudnessScale=1,
                         fnwave=None,onset=0):
    import scipy.signal as dsp
    import cochlea
    numCells = numCenterFreqs * 100 # should be ~100 * numCFs
    sampr = 100e3    
    # Make sound
    if fnwave=='chirp':
      duration = 1000
      toneFreq = 5000
      t = np.arange(0, duration/1000.0, 1/sampr)
      s = dsp.chirp(t, toneFreq-100, t[-1], toneFreq+100)
      s = cochlea.set_dbspl(s, 50 * loudnessScale)      
    elif fnwave:
      dat, sampr = readwav(fnwave)
      if sampr < 100e3:
        from scipy.signal import resample
        N = int(len(dat) * 100e3 / sampr)
        dat = resample(dat, N)
        sampr = 100000
      s = dat * loudnessScale
      # print('s range:',np.amin(s),np.amax(s))
    else:
      s = []
    # pad = np.zeros(int(10e-3 * sampr))
    sound = s # np.concatenate( (s, pad) )
    # print('sound range:',np.amin(sound),np.amax(sound))
    # Run model
    anf = cochlea.run_zilany2014(
        sound,
        sampr,
        anf_num=(numCells/numCenterFreqs, 0, 0),  # the desired number of auditory nerve fibers per frequency channel (CF)
        cf=(freqRange[0], freqRange[1], numCenterFreqs), # the center frequency(s) of the simulated auditory nerve fibers
        seed=0,
        powerlaw='approximate',
        species='human',
        ffGn=False
    )
    # generate list of spk times
    if len(s) > 0:
      spkTimes = [list(anf.iloc[i]['spikes']*1000. + onset) for i in range(numCells)]
    else:
      spkTimes = [[] for i in range(numCells)]
    cf = [anf.iloc[i]['cf'] for i in range(numCells)]    
    return {'spkT':spkTimes, 'cf':cf}


# def cochlearInputSpikesBrianHears(freqRange=[9000,11000], # orig: [20, 20000],
#                         numCells=200, # orig: [3000]
#                         duration=1000,
#                         toneFreq=10000,
#                         plotRaster=False): 

#     from brian2 import *
#     from brian2hears import *

#     print(' Generating cochlear-like auditory input spikes using Brian Hears ...')
#     cfmin, cfmax, cfN = freqRange[0]*Hz, freqRange[1]*Hz, numCells
#     cf = erbspace(cfmin, cfmax, cfN)
#     sound1 = tone(toneFreq*Hz, duration*ms)
#     sound2 = whitenoise(duration*ms)
#     sound = sound1+sound2
#     sound = sound.ramp()
#     gfb = Gammatone(sound, cf)
#     ihc = FunctionFilterbank(gfb, lambda x: 3 * clip(x, 0, Inf)**(1.0 / 3.0))

#     # Leaky integrate-and-fire model with noise and refractoriness
#     eqs = '''
#     dv/dt = (I-v)/(1*ms)+0.2*xi*(2/(1*ms))**.5 : 1 (unless refractory)
#     I : 1
#     '''
#     G = FilterbankGroup(ihc, 'I', eqs, reset='v=0', threshold='v>1', refractory=5*ms, method='euler')

#     # Run, and raster plot of the spikes
#     M = SpikeMonitor(G)
#     run(sound.duration)
#     if plotRaster:
#         plot(M.t / ms, M.i, '.')
#         plt.show()

#     # generate list of spk times
#     spkts = list(M.t)
#     spkids = list(M.i)
#     spkTimes = [[] for i in range(numCells)] 
#     for spkt, spkid in zip(spkts, spkids):
#         spkTimes[spkid].append(float(spkt)*1000.)

#     return spkTimes


def poisson_generator(rate, t_start=0.0, t_stop=1000.0, seed=None):
    """
    Returns a SpikeTrain whose spikes are a realization of a Poisson process
    with the given rate (Hz) and stopping time t_stop (milliseconds).

    Note: t_start is always 0.0, thus all realizations are as if 
    they spiked at t=0.0, though this spike is not included in the SpikeList.

    Inputs:
    -------
        rate    - the rate of the discharge (in Hz)
        t_start - the beginning of the SpikeTrain (in ms)
        t_stop  - the end of the SpikeTrain (in ms)
        array   - if True, a np array of sorted spikes is returned,
                    rather than a SpikeTrain object.

    Examples:
    --------
        >> gen.poisson_generator(50, 0, 1000)
        >> gen.poisson_generator(20, 5000, 10000, array=True)

    See also:
    --------
        inh_poisson_generator, inh_gamma_generator, inh_adaptingmarkov_generator
    """

    rng = np.random.RandomState(seed)

    #number = int((t_stop-t_start)/1000.0*2.0*rate)

    # less wasteful than double length method above
    n = (t_stop-t_start)/1000.0*rate
    number = np.ceil(n+3*np.sqrt(n))
    if number<100:
        number = min(5+np.ceil(2*n),100)

    if number > 0:
        isi = rng.exponential(1.0/rate, int(number))*1000.0
        if number > 1:
            spikes = np.add.accumulate(isi)
        else:
            spikes = isi
    else:
        spikes = np.array([])

    spikes+=t_start
    i = np.searchsorted(spikes, t_stop)

    extra_spikes = []
    if i==len(spikes):
        # ISI buf overrun
        
        t_last = spikes[-1] + rng.exponential(1.0/rate, 1)[0]*1000.0

        while (t_last<t_stop):
            extra_spikes.append(t_last)
            t_last += rng.exponential(1.0/rate, 1)[0]*1000.0
        
        spikes = np.concatenate((spikes,extra_spikes))

    else:
        spikes = np.resize(spikes,(i,))

    return spikes

def inh_poisson_generator(rate, t, t_stop, seed=None):
    """
    Returns a SpikeTrain whose spikes are a realization of an inhomogeneous 
    poisson process (dynamic rate). The implementation uses the thinning 
    method, as presented in the references.

    Inputs:
    -------
        rate   - an array of the rates (Hz) where rate[i] is active on interval 
                    [t[i],t[i+1]]
        t      - an array specifying the time bins (in milliseconds) at which to 
                    specify the rate
        t_stop - length of time to simulate process (in ms)
        array  - if True, a np array of sorted spikes is returned,
                    rather than a SpikeList object.

    Note:
    -----
        t_start=t[0]

    References:
    -----------

    Eilif Muller, Lars Buesing, Johannes Schemmel, and Karlheinz Meier 
    Spike-Frequency Adapting Neural Ensembles: Beyond Mean Adaptation and Renewal Theories
    Neural Comput. 2007 19: 2958-3010.

    Devroye, L. (1986). Non-uniform random variate generation. New York: Springer-Verlag.

    Examples:
    --------
        >> time = arange(0,1000)
        >> stgen.inh_poisson_generator(time,sin(time), 1000)

    See also:
    --------
        poisson_generator, inh_gamma_generator, inh_adaptingmarkov_generator
    """

    rng = np.random.RandomState(seed)

    if np.shape(t)!=np.shape(rate):
        raise ValueError('shape mismatch: t,rate must be of the same shape')

    # get max rate and generate poisson process to be thinned
    rmax = np.max(rate)
    ps = poisson_generator(rate=rmax, t_start=t[0], t_stop=t_stop, seed=seed)

    # return empty if no spikes
    if len(ps) == 0:
        np.array([])
        
    # gen uniform rand on 0,1 for each spike
    rn = np.array(rng.uniform(0, 1, len(ps)))

    # instantaneous rate for each spike
    idx = np.searchsorted(t, ps) - 1
    #spike_rate = rate[idx]
    spike_rate = np.array([rate[i] for i in idx])

    # thin and return spikes
    spike_train = ps[rn<spike_rate/rmax]

    return list(spike_train)
    


# main
#spk=cochlearInputSpikes(plotRaster=1)
