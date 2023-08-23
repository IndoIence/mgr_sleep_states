import os
import pandas as pd
import mne
import numpy as np
import sys

from matplotlib import pyplot as plt

edf_dir = r"Jean-Pol_repaired_headers"

stages_names = {
    0: 'Wake',
    1: 'N1',
    2: 'N2',
    3: 'N3',
    4: 'N4',
    5: 'REM'
}
stages_names_3_outputs = {
    0: 'Wake',
    1: 'NREM',
    2: 'REM'
}


def date_from_comment(file):
    """this can be just taken from mne info meas_date probably"""
    with open(file) as f:
        for line in f:
            if line.startswith("% night ="):
                return line[-11:-1]


def import_stages(name, path=edf_dir):
    all_files = os.listdir(path)
    stages_files = list(filter(lambda x: x[-10:] == 'stages.txt', all_files))
    return np.loadtxt(name)


def import_ecg(name, tmin=0, tmax=np.inf, path=edf_dir, ):
    """takes a name, returns a raw !!! if importing the CP229110.edf file, the data is flipped"""
    raw = mne.io.read_raw_edf(path + "//" + name)
    ecg_ind = mne.pick_channels_regexp(raw.info['ch_names'], 'ECG*')
    incl_ch = [raw.info['ch_names'][i] for i in ecg_ind]
    raw.pick_channels(incl_ch)
    if tmin == 0 and tmax == np.inf:
        return raw
    raw.crop(tmin=tmin, tmax=tmax)
    return raw


def get_edf_filenames(path=edf_dir):
    all_files = os.listdir(path)
    edf_files = list(filter(lambda x: x[-4:] == '.edf', all_files))
    edf_files = sorted(edf_files)
    # this removes _waking and such that are included in the directory
    edf_files = [name for name in edf_files if '_' not in name]
    return edf_files


def load_data_one_file(edf):
    """takes in an EDF or a string with edf name:
    returns np array of shape (nr_windows, 10000)"""
    if type(edf) == str:
        edf = import_ecg(edf)
    sampl_freq = edf.info["sfreq"]
    window_len = int(sampl_freq * 20)
    nr_windows = int(len(edf[0][1]) // window_len)
    print(f'{edf} with {nr_windows} windows')
    # one of the edfs is inverted, returning it with y flipped
    name = edf.filenames[0]
    # get just the filename without the path
    name = os.path.basename(name)
    y = edf[0][0].reshape(-1)
    y = y[0:nr_windows * window_len]
    y = y.reshape((-1, window_len))
    if name == 'CP229110.edf':
        print(f"importing inverted file: {name}")
        print(f"{edf.filenames[0]}")
        y = y*-1
    return y


def load_all_data():
    names = get_edf_filenames()
    # edfs = [import_ecg(f) for f in names[:2]]
    edfs = [import_ecg(f) for f in names]
    all_data = np.array([])
    for edf in edfs:
        y = load_data_one_file(edf)
        all_data = np.append(all_data, y)

    return all_data.reshape((-1, int(1e4)))


def load_all_labels():
    edfs_names = get_edf_filenames()
    all_stages = []
    for name in edfs_names:
        fname = edf_dir + '//' + name + '_stages.txt'
        print(f'loading from {fname}')
        stages = pd.read_csv(fname, comment='%', delimiter='	', index_col=0).drop('Unnamed: 3', axis=1)
        raw_stages = np.copy(stages['stage'])
        all_stages.extend(raw_stages)
    return np.array(all_stages)


def load_labels_one_file(name):
    fname = edf_dir + '//' + name + '_stages.txt'
    print(f'loading from {fname}')
    stages = pd.read_csv(fname, comment='%', delimiter='	', index_col=0).drop('Unnamed: 3', axis=1)
    return np.copy(stages['stage'])


def sizeof(obj):
    size = sys.getsizeof(obj)
    if isinstance(obj, dict): return size + sum(map(sizeof, obj.keys())) + sum(map(sizeof, obj.values()))
    if isinstance(obj, (list, tuple, set, frozenset)): return size + sum(map(sizeof, obj))
    return size


def three_stages_transform(l):
    def helper(n):
        if n == 0:
            return 0
        if n == 5:
            return 2
        return 1

    return list(map(helper, l))


def two_stages_transform(l):
    def helper(n):
        if n == 0:
            return 0
        return 1

    return list(map(helper, l))


def remove_ecg_artifacts(data, labels=None, threshold=0.003):
    # removes all 20s datapoints and their labels from the pool if there is a point over certain threshold
    ecg_artifact_filter = np.all(np.abs(data) <= threshold, axis=1)
    if labels is not None:
        labels = labels[ecg_artifact_filter]
    data = data[ecg_artifact_filter]
    return data, labels


def draw_hypnogram(y, y_stages, name=None):
    assert len(y) == len(y_stages)
    fig, ax1 = plt.subplots(figsize=(20, 6))
    ax2 = ax1.twinx()
    ax1.step(y, 'b-')
    ax2.step(y_stages, 'g-')
    ax2.invert_yaxis()
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('Voltage [V]', color='b')
    ax2.set_ylabel('Sleep Stage', color='g')
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # plt.legend(['ECG'])
    if name is not None:
        plt.title(f"ECG of {name}")
    return ax1
