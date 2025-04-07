#! /use/bin/env python

import numpy as np
from scipy.fftpack import next_fast_len
from obspy.signal.filter import bandpass
from obspy import Trace
from obspy.signal.filter import bandpass
from obspy.geodetics import locations2degrees
# import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import threading
import time
# from memory_profiler import profile

def process2spec(trace, params):
    """
    Single station processing of a waveform trace and return processed spectrum.
    The processing includes temporal and spectral whitening.

    :param trace: ObsPy Trace object
    :return: processed spectrum of the input trace

    Global parameters used in this function:
    :param temp_width: window width of the running absolute-mean temporal normalization
        if temp_width is None, no temporal whitening is applied
    :param ram_fband: pre-filter period band used in the computation of the normalization weight
        if ram_fband is None, no pre-filtering is applied
    :param spec_width: width of the running asbolute-mean spectral normalization
        if spec_width is None, no spectral whitening is applied
    """
    if type(trace) is not Trace:
        raise TypeError('Input trace is not an ObsPy Trace object.')
    
    # number of desired points in the waveform
    delta = 1. #/ params['preprocess']['resample_rate']
    npts = 7*3600 #int((params['sliding_window']['length']) / delta)

    ## --- STEP 1: temporal whitening
    # return orignal data if temp_width is None
    # compute normalizing weight (filter if specified) [Bensen et al., 2007]
    filtered = trace.data.copy()
    ram_fband = (0.02, 0.067)
    if ram_fband is not None:
        # filter the original trace to compute weight
        filtered = bandpass(filtered, df=1.0/delta, zerophase=True, \
                freqmin=ram_fband[0], freqmax=ram_fband[1], corners=4)
    # smoothen by convolving with an average mask
    temp_width = 128
    winlen = 2 * int(0.5 * temp_width / delta) + 1
    avg_mask = np.ones(winlen) / (1.0 * winlen)
    filtered = np.convolve(np.abs(filtered), avg_mask, 'same')
    # divide the orignal data by the smoothed weight
    mask = (filtered > 1e-8*np.max(np.abs(filtered)))
    trace.data[mask] /= filtered[mask]
    trace.data[np.logical_not(mask)] = 0
    # taper the trace by 1% cosine taper
    trace.taper(type='cosine', max_percentage=0.01) # inplace

    ## --- STEP 2: spectral whitening
    # get the number of points for FFT using tensorflow fft module
    fft_npts = next_fast_len(2*npts)
    # real-to-complex FFT by numpy
    spec_data = np.fft.rfft(trace.data, fft_npts)
    
    # if spec_width is None, return the orignial spectrum
    spec_width = 2e-3 #params['spec_norm']['width']
    # create convolution kernel to smooth the spectrum using average filter
    winlen = 2 * int(0.5 * spec_width * (delta * fft_npts)) + 1
    avg_filter = np.ones(winlen) / (1.0 * winlen)
    weight = np.convolve(np.abs(spec_data), avg_filter, 'same')
    # divide the orignal data by the smoothed weight
    mask = (weight > 1e-8*np.max(np.abs(weight)))
    spec_data[mask] /= weight[mask]
    spec_data[np.logical_not(mask)] = 0j

    # return the whitened spectrum
    return spec_data

def single_station_processing(dstream, params, pool=None):
    '''
    Apply single station processing (in parallel) of a waveform stream and 
    return processed spectrum.
    '''
    if pool is None:
        spec_data = np.array([process2spec(trace, params) for trace in dstream])
    else:
        args = [(trace, params) for trace in dstream]
        spec_data = np.array(pool.starmap(process2spec, args))
    return spec_data

def alloc2bins(stlats, stlons, bins):
    '''
    Calculate the bin index for each inter-receiver distance.

    Args:
        stlats (ndarray): The latitude of the receivers.
        stlons (ndarray): The longitude of the receivers.
        bin_size (float): The size of the bin in degrees.

    Returns:
        ndarray: The bin index for each inter-receiver distance.
    '''
    # calculate all inter-receiver distances
    mlats, mlons = np.meshgrid(stlats, stlons)
    cc_gcarc = locations2degrees(mlats, mlons, mlats.T, mlons.T)
    # determine the bin index for each inter-receiver distance
    cc_inds = np.digitize(cc_gcarc, bins) - 1
    return cc_inds

def xcorr_stack(spec_data, nbins, cc_inds, nmaxlag, ncores=2):
    '''
    Calculates the cross-correlation function of the given spectral data. Run the jobs
    on a GPU device if available, otherwise, run on CPU using threading.

    Args:
        spec_data (ndarray): The spectral data, represented as a complex numpy array.
        nbins (int): The number of cross-correlation distance bins.
        cc_inds (ndarray): The indices of the cross-correlation bins.

    Returns:
        ndarray: The cross-correlation function in the time domain.
    '''
    # number of traces and frequency bins
    ntraces, nspec = spec_data.shape

    if False:# tf.config.list_physical_devices('GPU'): # Check if a GPU device is available
        # Convert to tensorflow constant to speed up the calculation
        # by avoiding the data copy from CPU to GPU multiple times
        spec_data = tf.constant(spec_data, dtype=tf.complex128)
        cc_inds = tf.constant(cc_inds, dtype=tf.int32)

        # Initialize variables
        spec_xcorr_stack = tf.Variable(tf.zeros([nbins, nspec], dtype=tf.complex128))
        bin_count = tf.Variable(tf.zeros(nbins, dtype=tf.float32))

        for s1 in range(ntraces):
            # Cross-correlation function in frequency domain
            spec_xcorr = tf.math.conj(spec_data[s1]) * spec_data[s1:]
            
            # One-hot matrix of cross-correlation pairs for stacking
            onehot = tf.one_hot(cc_inds[s1, s1:], depth=nbins, dtype=tf.float32, axis=0)
            onehot_complex = tf.cast(onehot, dtype=tf.complex128)

            # Bin pair cross-correlograms in frequency domain
            spec_xcorr_stack = spec_xcorr_stack + tf.linalg.matmul(onehot_complex, spec_xcorr)

            # Count the trace pairs in each inter-receiver bin
            bin_count = bin_count + tf.reduce_sum(onehot, axis=1)

        # inverse FFT to get the cross-correlation function in time domain
        corrwf = tf.signal.irfft(spec_xcorr_stack).numpy()
        bin_count = bin_count.numpy()
    else:
        # global variables
        spec_xcorr_stack = np.zeros([nbins, nspec], dtype=np.complex128)
        bin_count = np.zeros(nbins, dtype=np.float32)

        # Create a lock for each bin
        # Note: the multiple locks for each bin is recommended to avoid congested
        # memory access when multiple threads are trying to access the same bin.
        locks = [threading.Lock() for _ in range(nbins)]

        # Local cross-correlation function to excecuted in multiple threads
        def cross_correlation(s1):
            # Cross-correlation function in frequency domain
            spec_xcorr = np.conj(spec_data[s1]) * spec_data[s1:]
            for _, ind in enumerate(cc_inds[s1, s1:]):
                # Acquire the lock for stacking
                with locks[ind]:
                    # Bin pair cross-correlograms in frequency domain
                    spec_xcorr_stack[ind] += spec_xcorr[_]
                    # Count the trace pairs in each inter-receiver bin
                    bin_count[ind] += 1

        # Create a ThreadPoolExecutor
        with ThreadPoolExecutor(ncores) as executor:
            for s1 in range(ntraces):
                # Run the cross-correlation function in a separate thread
                executor.submit(cross_correlation, s1)
        
        # inverse FFT to get the cross-correlation function in time domain
        corrwf = np.fft.irfft(spec_xcorr_stack)

    # fold the cross-correlation function and trim the time window
    corrwf = 0.5 * (corrwf + corrwf[:, ::-1])[:, :nmaxlag]

    return corrwf, bin_count

def write_corrwf(corrwf, bin_count, delta, bin_size, fname):
    '''
    Write the cross-correlation function to a h5 file.

    Args:
        corrf (ndarray): The cross-correlation function in frequency domain.
        bin_count (ndarray): The number of trace pairs in each bin.
        delta (float): The sampling rate of the waveform data.
        bin_size (float): The size of the bin in degrees.
        fname (str): Name of the h5 output file.
    '''
    from h5py import File
    with File(fname, 'w' ) as fp:
        fp.attrs['delta'] = delta
        fp.attrs['bin_size'] = bin_size
        dset1 = fp.create_dataset('corrwf', data=corrwf)
        dset2 = fp.create_dataset('bin_count', data=bin_count)
        fp.close()

# @profile
def xcorrelate(spec_data, stat_names, nmaxlag, buffer_size=64):
    '''
    Calculates the cross-correlation function of the given spectral data. Run the jobs
    on a GPU device if available, otherwise, run on CPU using threading.

    Args:
        spec_data (ndarray): The spectral data, represented as a complex numpy array.
        nmaxlag (int): The maximum lag of the cross-correlation function.
        stat_names (list): The list of station names corresponding to spec_data.

    Returns:
        ndarray: The cross-correlation function in the time domain.
    '''
    # number of traces and frequency bins
    ntraces, nspec = spec_data.shape
    row, col = np.triu_indices(ntraces)

    # cross-correlate in frequency domain
    xcorr = np.zeros([len(row), 2*nmaxlag+1], dtype=np.float32)
    
    # support function for parallel processing
    def __xcorr(start, end):
        spec_xcorr = np.conj(spec_data[row[start:end]]) * spec_data[col[start:end]]
        tmp = np.fft.irfft(spec_xcorr)
        xcorr[start:end] = np.fft.fftshift(tmp)[:, (nspec-nmaxlag):(nspec+nmaxlag+1)]

    # run the cross-correlation serially
    for _s in range(0, len(row), buffer_size):
        _e = min(_s+buffer_size, len(row))
        __xcorr(_s, _e)
    
    # Generate corresponding station name pairs
    xcorr_names = np.array(['_'.join([stat_names[i], stat_names[j]]) for i, j in zip(row, col)])    
    return xcorr, xcorr_names