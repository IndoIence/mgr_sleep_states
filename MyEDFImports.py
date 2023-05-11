import os

import mne
import numpy as np

edf_dir = r"Jean-Pol_repaired_headers"

def import_stages(name, path=edf_dir):
    all_files = os.listdir(path)
    stages_files = list(filter(lambda x: x[-10:] == 'stages.txt', all_files))
    return np.loadtxt(name)

def import_ecg(name, tmin, tmax, path=edf_dir, ):
    raw = mne.io.read_raw_edf(path + "//" + name)
    ecg_ind = mne.pick_channels_regexp(raw.info['ch_names'], 'ECG*')
    incl_ch = [raw.info['ch_names'][i] for i in ecg_ind]
    raw.pick_channels(incl_ch)
    raw.crop(tmin=tmin, tmax=tmax)
    return raw


def import_eeg(name, tmin, tmax, path=edf_dir, ):
    raw = mne.io.read_raw_edf(path + "//" + name)
    incl_ch = ["FP1A2 EEG"]
    raw.pick_channels(incl_ch)
    raw.crop(tmin=tmin, tmax=tmax)
    return raw


def get_edf_filenames(path=edf_dir):
    all_files = os.listdir(path)
    edf_files = list(filter(lambda x: x[-4:] == '.edf', all_files))
    return edf_files
