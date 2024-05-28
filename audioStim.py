#!/usr/bin/env python
# coding: utf-8

# written by Chase Mackey, samn, and Scott McElroy

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.io.wavfile import write
import sounddevice as sd
import matplotlib.pyplot as plt

def bandpass_filter (data, lowcut, highcut, fs, order=5):
  b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
  return filtfilt(b, a, data)

def add_ramps (data, ramp_duration, fs):
  ramp_samples = int(ramp_duration * fs)
  ramp_up = np.linspace(0, 1, ramp_samples)
  ramp_down = np.linspace(1, 0, ramp_samples)
  data[:ramp_samples] *= ramp_up
  data[-ramp_samples:] *= ramp_down
  return data

def generate_sine_wave_noise (duration, fs, center_freq, width=1/3, ramp_duration=0.01):
  # Calculate lowcut and highcut for 1/3 octave
  factor = 2 ** (width / 2)
  lowcut = center_freq / factor
  highcut = center_freq * factor    
  # Time axis
  t = np.linspace(0, duration, int(fs * duration), endpoint=False)    
  # Initialize noise
  noise = np.zeros_like(t)    
  # Generate 2000 sine waves with random frequencies and phases within the band
  for _ in range(2000):
      freq = np.random.uniform(lowcut, highcut)
      phase = np.random.uniform(0, 2 * np.pi)
      sine_wave = np.sin(2 * np.pi * freq * t + phase)
      noise += sine_wave    
  # Apply ramps before normalizing
  noise = add_ramps(noise, ramp_duration, fs)    
  # Normalize the noise to have a peak-to-peak amplitude of -1 to 1 after applying ramps
  noise /= np.max(np.abs(noise))    
  return noise

def generate_intervals (repetition_rate, jitter, duration, fs):
  mean_interval = 1 / repetition_rate
  intervals = np.abs(np.random.normal(mean_interval, jitter, int(duration / mean_interval)))
  return intervals

# Function to generate a click
def generate_click (click_duration, n_click_samples, frequency, amplitude):
  t = np.linspace(0, click_duration, n_click_samples, False)
  click = np.sin(frequency * t * 2 * np.pi)
  return amplitude * click

# Function to generate a BBN (just normally distributed noise * amplitude)
def generate_BBN (sampr, dur, amp, seed=1234):
  # sampr = sampling rate
  # dur = duration in seconds
  # amp = amplitude
  np.random.seed(seed)
  return amp * np.random.normal(0,1,int(sampr*dur))

# Function to generate silence
def generate_silence (duration_samples): return np.zeros(duration_samples)  

def generate_clicktrain (sampr=44100.0, duration=3.0, clickduration=0.5, silenceduration=1.0, freq=40.0, amplitude=500,initsilence=3):
  # Parameters: sampr=sample rate in Hz, duration=duration of full train seconds, clickduration=duration of a single click
  # silenceduration=duration of silent period (between clicks) in seconds
  # freq=frequency of click train in Hz, amplitude=amplitude of click
  # initisilence=number of seconds of silence before clicks begin
  n_samples = int(sampr * duration)
  n_click_samples = int(sampr * clickduration)
  n_silence_samples = int(sampr * silenceduration)
  n_initial_silence_samples = int(sampr * initsilence)  # seconds of initial silence
  # Generate the click train and silence
  click_train = generate_click(clickduration, n_click_samples, freq, amplitude)
  silence = generate_silence(n_silence_samples)
  # initial_silence = generate_silence(n_initial_silence_samples)
  # Combine the initial silence, click train and silence to create the desired pattern
  pattern = np.concatenate(( click_train, silence, click_train, silence))
  # Repeat the pattern to fill the duration of the .wav file
  n_patterns = int(n_samples / len(pattern))
  sound = np.tile(pattern, n_patterns)
  # Ensure the sound array is within the correct range for a .wav file
  sound = np.clip(sound, -1, 1)
  # Convert to 16-bit data
  # sound = (sound * 32767).astype(np.int16)
  return sound

def generate_BBNTrain (sampr=44100.0, duration=3.0, bbnduration=0.1, silenceduration=1.0, amplitude=1,initsilence=3):
  # Parameters: sampr=sample rate in Hz, duration=duration of full train seconds, BBNduration=duration of a single BBN
  # silenceduration=duration of silent period (between clicks) in seconds
  # initisilence=number of seconds of silence before clicks begin
  n_samples = int(sampr * duration)
  n_silence_samples = int(sampr * silenceduration)
  n_initial_silence_samples = int(sampr * initsilence)  # seconds of initial silence
  # Generate the click train and silence
  bbn_train = generate_BBN(sampr, bbnduration, amplitude)
  silence = generate_silence(n_silence_samples)
  # initial_silence = generate_silence(n_initial_silence_samples)
  # Combine the initial silence, bbn train and silence to create the desired pattern
  pattern = np.concatenate(( bbn_train, silence, bbn_train, silence))
  # Repeat the pattern to fill the duration of the .wav file
  n_patterns = int(n_samples / len(pattern))
  sound = np.tile(pattern, n_patterns)
  # Ensure the sound array is within the correct range for a .wav file
  # sound = np.clip(sound, -1, 1)
  # Convert to 16-bit data
  # sound = (sound * 32767).astype(np.int16)
  return sound  

def plot_fft (data, fs, xmin=20, xmax=5000):  # Adjusted xmin to 20 Hz for logarithmic scale
  fft_data = np.fft.fft(data)
  freqs = np.fft.fftfreq(len(fft_data), 1/fs)
  half_len = len(freqs) // 2  # Only plotting the positive frequencies
  # Filtering out the frequencies and FFT data for positive frequencies and within xmin and xmax
  valid_indices = np.where((freqs > 0) & (freqs <= xmax))
  valid_freqs = freqs[valid_indices][:half_len]
  valid_fft_data = np.abs(fft_data)[valid_indices][:half_len]
  plt.plot(valid_freqs, valid_fft_data)
  plt.title('FFT')
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Amplitude')
  plt.xlim(xmin, xmax)
  plt.xscale('log')  # Set the x-axis to a logarithmic scale
  plt.show()

def create_sound_stream (center_freq, repetition_rate, jitter=0, duration=10, noise_duration=0.1, fs=44100, rhythmic=True, asList=False):
  intervals = generate_intervals(repetition_rate, jitter, duration, fs) if not rhythmic else np.full(int(duration * repetition_rate), 1 / repetition_rate)    
  sound_stream = np.array([])
  lnoise = []
  for interval in intervals:
      # Generate noise burst of specified duration
      noise = generate_sine_wave_noise(noise_duration, fs, center_freq, width=1/3, ramp_duration=0.005)
      lnoise.append(noise)
      # Add silence after the noise burst until the next interval starts
      silence_duration = int((interval - noise_duration) * fs) if interval > noise_duration else 0
      silence = np.zeros(silence_duration)
      sound_stream = np.concatenate((sound_stream, noise, silence))    
  # Normalize to prevent clipping
  divisor = np.max(np.abs(sound_stream))
  sound_stream = sound_stream / divisor
  for i in range(len(lnoise)): lnoise[i] /= divisor
  if asList:
    return lnoise, intervals, sound_stream
  else:
    return sound_stream

def testJitterNoise (fn='output_sound.wav',fs=44100):
  plt.ion()
  # Example usage
  # fs = 44100  # Sampling frequency
  center_freq = 1000  # Center frequency of the noise
  repetition_rate = 1.6  # Repetition rate in Hz
  jitter = 0.1  # Standard deviation of the intervals in Sec., set to 0 for perfectly rhythmic. We used 0, 0.04,0.08,0.12. 
  duration = 10  # Total duration of the sound stream in seconds
  noise_duration = 0.025  # Duration of each noise burst in seconds
  sound_stream = create_sound_stream(center_freq, repetition_rate, jitter, duration, noise_duration, fs, rhythmic=False)
  # Play sound
  sd.play(sound_stream, fs)
  sd.wait()
  # Save to file
  write(fn, fs, sound_stream.astype(np.float32))
  # check spectrum of the noise
  plot_fft(sound_stream[:int(fs*noise_duration)], fs,100,4000)
  # check stream of noises
  time = np.arange(0, len(sound_stream)) / fs
  plt.figure(figsize=(10, 4))
  plt.plot(time, sound_stream)
  plt.title('Audio Stream')
  plt.xlabel('Time (s)')
  plt.ylabel('Amplitude')
  plt.show()
  # Play sound
  #sd.play(sound_stream, fs)
  #sd.wait()

def testClickTrain (fn='clicktrain.wav',fs=44100,clickduration=0.5):
  plt.ion()
  sound_stream = generate_clicktrain(clickduration=clickduration)
  # Save to file
  write(fn, fs, sound_stream.astype(np.float32))
  # check spectrum of the noise
  plot_fft(sound_stream[:int(fs*clickduration)], fs,100,4000)
  # check stream of noises
  time = np.arange(0, len(sound_stream)) / fs
  plt.figure(figsize=(10, 4))
  plt.plot(time, sound_stream)
  plt.title('Click Train Audio Stream')
  plt.xlabel('Time (s)')
  plt.ylabel('Amplitude')
  plt.show()
  # Play sound
  #sd.play(sound_stream, fs)
  #sd.wait()

def testBBNTrain (fn='bbntrain.wav',fs=44100,bbnduration=0.1,amplitude=1):
  plt.ion()
  sound_stream = generate_BBNTrain(bbnduration=bbnduration,amplitude=amplitude)
  # Save to file
  write(fn, fs, sound_stream.astype(np.float32))
  # check spectrum of the noise
  plot_fft(sound_stream[:int(fs*bbnduration)], fs,100,4000)
  # check stream of noises
  time = np.arange(0, len(sound_stream)) / fs
  plt.figure(figsize=(10, 4))
  plt.plot(time, sound_stream)
  plt.title('BBN Train Audio Stream')
  plt.xlabel('Time (s)')
  plt.ylabel('Amplitude')
  plt.show()
  # Play sound
  #sd.play(sound_stream, fs)
  #sd.wait()

def testBBN (fn='bbn.wav',fs=44100,bbnduration=0.1,amplitude=1):
  plt.ion()
  sound_stream = generate_BBN(fs, bbnduration, amplitude) 
  # Save to file
  write(fn, fs, sound_stream.astype(np.float32))
  # check spectrum of the noise
  plot_fft(sound_stream[:int(fs*bbnduration)], fs,100,4000)
  # check stream of noises
  time = np.arange(0, len(sound_stream)) / fs
  plt.figure(figsize=(10, 4))
  plt.plot(time, sound_stream)
  plt.title('BBN Audio Stream')
  plt.xlabel('Time (s)')
  plt.ylabel('Amplitude')
  plt.show()
  # Play sound
  #sd.play(sound_stream, fs)
  #sd.wait()  
  

if __name__ == '__main__':
  import sys
  print(sys.argv)
  if len(sys.argv) > 1:
    if sys.argv[1] == 'jitter':
      testJitterNoise()
    elif sys.argv[1] == 'clicktrain':
      testClickTrain()
    elif sys.argv[1] == 'bbntrain':
      testBBNTrain()
    elif sys.argv[1] == 'bbn':
      testBBN()
    elif sys.argv[1] == 'silence':
      testBBNTrain(fn='silence.wav',amplitude=0.0)


