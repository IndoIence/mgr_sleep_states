# -*- coding: utf-8 -*-
"""
Author: CHANDAN ACHARYA.
Date : 1 May 2019.
"""
########################### LIBRARIES #########################################
from __future__ import division

import statistics

from matplotlib import pyplot as plt
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.interpolate import interp1d
from scipy.signal import welch

import MyEDFImports as m

####################### FEATURE DEFINITIONS ###################################
"""TIME DOMAIN"""


# independent function to calculate RMSSD
def calc_rmssd(data_list):
    diff_nni = np.diff(data_list)  # successive differences
    return np.sqrt(np.mean(diff_nni ** 2))


# independent function to calculate AVRR
def calc_avrr(data_list):
    return sum(data_list) / len(data_list)


# independent function to calculate SDRR
def calc_sdrr(data_list):
    return np.std(data_list)


# independent function to calculate SKEW
def calc_skew(list):
    return skew(list)


# independent function to calculate KURT
def calc_kurt(list):
    return kurtosis(list)


def calc_NNx(list):
    diff_nni = np.diff(list)
    return sum(np.abs(diff_nni) > 50)


def calc_pNNx(list):
    length_int = len(list)
    diff_nni = np.diff(list)
    nni_50 = sum(np.abs(diff_nni) > 50)
    return 100 * nni_50 / length_int


"""NON LINEAR DOMAIN"""


# independent function to calculate SD1
def calc_SD1(list):
    diff_nn_intervals = np.diff(list)
    return np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)


# independent function to calculate SD2
def calc_SD2(list):
    diff_nn_intervals = np.diff(list)
    return np.sqrt(2 * np.std(list, ddof=1) ** 2 - 0.5 * np.std(
        diff_nn_intervals, ddof=1) ** 2)


# independent function to calculate SD1/SD2
def calc_SD1overSD2(list):
    diff_nn_intervals = np.diff(list)
    sd1 = np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(list, ddof=1) ** 2 - 0.5 * np.std(
        diff_nn_intervals, ddof=1) ** 2)
    ratio_sd2_sd1 = sd2 / sd1
    return ratio_sd2_sd1


# independent function to calculate CSI
def calc_CSI(list):
    diff_nn_intervals = np.diff(list)
    sd1 = np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(list, ddof=1) ** 2 - 0.5 * np.std(
        diff_nn_intervals, ddof=1) ** 2)
    L = 4 * sd1
    T = 4 * sd2
    return L / T


# independent function to calculate CVI
def calc_CVI(list):
    diff_nn_intervals = np.diff(list)
    sd1 = np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(list, ddof=1) ** 2 - 0.5 * np.std(
        diff_nn_intervals, ddof=1) ** 2)
    L = 4 * sd1
    T = 4 * sd2
    return np.log10(L * T)


# independent function to calculate modified CVI
def calc_modifiedCVI(list):
    diff_nn_intervals = np.diff(list)
    sd1 = np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(list, ddof=1) ** 2 - 0.5 * np.std(
        diff_nn_intervals, ddof=1) ** 2)
    L = 4 * sd1
    T = 4 * sd2
    return L ** 2 / T




# sliding window function
def slidingWindow(sequence, winSize, step):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""

    # Verify the inputs
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence\
                        length.")
    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence) - winSize) / step) + 1
    # Do the work
    for i in range(0, int(numOfChunks) * step, step):
        yield sequence[i:i + winSize]


####################### FEATURE EXTRACTION ####################################

import numpy as np
from functools import partial


# Define the base feature calculation functions

def calculate_percentiles(data, mode, percentile):
    if mode == "detrended":
        data = detrend(data)
    elif mode == "absolute":
        data = absolute(data)
    return np.percentile(data, percentile)


def detrend(data):
    linear_fit = np.polyfit(range(len(data)), data, 1)
    trend = np.polyval(linear_fit, range(len(data)))
    return data - trend



def compute_hrv_metrics(rr_intervals, fs_resample=4.0):
    """
    Compute VLF, LF, HF powers and LF-to-HF ratio from RR intervals.

    Parameters:
    - rr_intervals: List of RR intervals in seconds.
    - fs_resample: Resampling frequency (default is 4 Hz).

    Returns:
    - Dictionary containing VLF, LF, HF powers and LF-to-HF ratio.
    """

    # Time array
    time_array = np.cumsum(rr_intervals)

    # Interpolate to get evenly spaced values
    time_interpolated = np.arange(0, time_array[-1], 1 / fs_resample)
    interpolated_function = interp1d(time_array, rr_intervals, kind='cubic')
    rr_interpolated = interpolated_function(time_interpolated)

    # Compute the power spectral density using Welch's method
    f, psd = welch(rr_interpolated, fs=fs_resample, nperseg=256, noverlap=128)

    # Integrate the power in the VLF, LF, and HF bands
    vlf_power = np.trapz(psd[(f >= 0.0033) & (f < 0.04)])
    lf_power = np.trapz(psd[(f >= 0.04) & (f < 0.15)])
    hf_power = np.trapz(psd[(f >= 0.15) & (f < 0.4)])

    # Compute LF-to-HF ratio
    lf_hf_ratio = lf_power / hf_power

    return {
        "VLF": vlf_power,
        "LF": lf_power,
        "HF": hf_power,
        "LF/HF": lf_hf_ratio
    }


def absolute(data):
    return np.abs(data)


def get_all_functions_dict():
    feature_funcs = {
        "RMSSD": calc_rmssd,
        "AVRR": calc_avrr,
        "SDRR": calc_sdrr,
        "SKEW": calc_skew,
        "KURT": calc_kurt,
        "NNx": calc_NNx,
        "pNNx": calc_pNNx,
        "SD1": calc_SD1,
        "SD2": calc_SD2,
        "SD1/SD2": calc_SD1overSD2,
        "CSI": calc_CSI,
        "CVI": calc_CVI,
        "modifiedCVI": calc_modifiedCVI,
        "HRV_metrics": compute_hrv_metrics
    }
    # Add percentile functions directly to the feature_funcs dictionary
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    for p in percentiles:
        feature_funcs[f"Detrended_{p}%"] = partial(calculate_percentiles, mode="detrended", percentile=p)
        feature_funcs[f"Absolute_{p}%"] = partial(calculate_percentiles, mode="absolute", percentile=p)
    return feature_funcs


def feature_extract(window_stream, features):
    result = {feature: [] for feature in features}
    feature_funcs = get_all_functions_dict()

    for window in window_stream:
        for feature in features:
            if feature in feature_funcs:
                result[feature].append(feature_funcs[feature](window))
            else:
                raise ValueError(f"{feature}: no such feature")

    return result


########################### PLOTTING ##########################################
def plot_features(featureList, label):
    plt.title(label)
    plt.plot(featureList)
    plt.show()


###################### CALLING FEATURE METHODS ################################
def browsethroughSeizures(list_rri, winSize, step):
    features = ["RMSSD", "AVRR", "SDRR", "SKEW", "KURT", "NNx", "pNNx", "SD1", "SD2",
                "SD1/SD2", "CSI", "CVI", "modifiedCVI"]
    for item in features:
        featureList = feature_extract(list_rri, winSize, step, item)
        plot_features(featureList, item)


#################### BAYESIAN CHANGE POINT DETECTION ##########################
####inspired by https://github.com/hildensia/bayesian_changepoint_detection
def bayesianOnFeatures(list_rri, winSize, step):
    features = ["RMSSD", "AVRR", "SDRR", "SKEW", "KURT", "NNx", "pNNx", "SD1", "SD2",
                "SD1/SD2", "CSI", "CVI", "modifiedCVI"]
    for item in features:
        featureList = feature_extract(list_rri, winSize, step, item)
        featureList = np.asanyarray(featureList)
        Q, P, Pcp = ocpd.offline_changepoint_detection \
            (featureList, partial(ocpd.const_prior, l=(len(featureList) + 1))
             , ocpd.gaussian_obs_log_likelihood, truncate=-40)
        fig, ax = plt.subplots(figsize=[15, 7])
        ax = fig.add_subplot(2, 1, 1)
        ax.set_title(item)
        ax.plot(featureList[:])
        ax = fig.add_subplot(2, 1, 2, sharex=ax)
        ax.plot(np.exp(Pcp).sum(0))


#################### CHANGE POINT DETECTION ##########################



    '''get the rolling mean and also plot the data'''

    # df = pd.DataFrame(data)
    # df
    # RM = df.rolling(window=30).mean().dropna()
    # fig, ax = plt.subplots(figsize=[18, 16])
    # ax = fig.add_subplot(2, 1, 1)
    # ax.plot(df)
    # ax = fig.add_subplot(2, 1, 2, sharex=ax)
    # ax.plot(RM)
    # rm = RM.values  # convert df to list
    # plt.show()
# In[]
