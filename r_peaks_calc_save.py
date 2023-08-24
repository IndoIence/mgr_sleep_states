import csv
import os
from functools import reduce

import matplotlib.pyplot as plt
import neurokit2 as nk2
import numpy as np
from scipy.signal import convolve

import MyEDFImports as m

s_freq = 500


def load_data_labels(names):
    raw_data_all = []
    labels_all = []
    for edf_filename in names:
        edf_labels = m.load_labels_one_file(edf_filename)
        raw = m.import_ecg(edf_filename)
        data = raw[0][0].flatten()
        raw_data_all.append(data)
        labels_all.append(edf_labels)

    return raw_data_all, labels_all


def make_windows(data_list, labels_list, win_pad):
    """win_pad -> number of windows taken from left and right to the labeled window"""
    assert len(data_list) == len(labels_list)
    assert win_pad > 0 and type(win_pad) == int
    win_len = (2 * win_pad + 1) * 500 * 20
    twenty_sec = 20 * 500
    for sleep_data, labels in zip(data_list, labels_list):
        n_wide_wind = len(labels) - win_pad * 2
        # cutting last few unlabeled seconds
        labeled_data = sleep_data[:twenty_sec * len(labels)]
        print(len(labeled_data))
        middle_labels = labels[win_pad:n_wide_wind + win_pad]
        for i, label in enumerate(middle_labels):
            start = i * twenty_sec
            stop = i * twenty_sec + win_len
            # print(start, stop, len(labeled_data[start:stop]), label)
            yield labeled_data[start:stop], label


def calc_r_peaks(windows_stream):
    for one_window, label in windows_stream:
        try:
            peaks = nk2.ecg_peaks(one_window.flatten(), sampling_rate=s_freq)[1]['ECG_R_Peaks']
        except:
            peaks = []
        yield one_window, peaks, label


def where_not_30bpm(peaks_list, len_data):
    " not working here"
    nr_seconds = len_data / 500
    where_less_than_half = np.array([len(p) > (nr_seconds / 2) for p in peaks_list])
    prcnt_discarded = sum(where_less_than_half) / len(where_less_than_half)

    print(prcnt_discarded * 100)
    return where_less_than_half


def del_from_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        return
    assert dir == 'r_peaks'
    try:
        files = os.listdir(dir)
        for file in files:
            # deleting only short filenames
            if len(file) <= 5:
                file_path = os.path.join(dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
    except OSError:
        print(f"Error deleting in {dir}")


def save_to_csv(windows_r_peaks_stream):
    csv_writers = {}
    open_files = {}
    # i should append for every new peak_list, but delete at first the file
    for win, peaks, label in windows_r_peaks_stream:
        if label not in csv_writers:
            file = open(f'r_peaks/{label}.csv', 'a', newline='')
            writer = csv.writer(file)
            csv_writers[label] = writer
            open_files[label] = file
        else:
            writer = csv_writers[label]
        writer.writerow(peaks)
    for f in open_files.values():
        f.close()


def load_r_peaks_csv(dir):
    files = os.listdir(dir)
    peaks = {}
    for file in files:
        lists = []
        if file[-4:] == '.csv':
            f_name = dir + f'/{file}'
            with open(f_name) as f:
                reader = csv.reader(f)
                for row in reader:
                    lists.append([int(value) for value in row])
        peaks[int(file[:-4])] = lists
    return peaks


def get_variance(data, nr_seconds):
    win_len = int(nr_seconds * 500)
    var_win = np.ones(win_len) / win_len
    mov_avg = convolve(data, var_win, mode='same')
    variance = convolve(data ** 2, var_win, mode='same') - mov_avg ** 2
    return variance

def variance_over_threshold(windowed_stream, nr_seconds, threshold):
    for window, label in windowed_stream:
        var = get_variance(window, nr_seconds)
        yield window, label, np.any(var >= threshold)


if __name__ == '__main__':
    edf_filenames = m.get_edf_filenames()[:1]
    # get data for every patient in a list of np arrays
    data, labels_init = load_data_labels(edf_filenames)
    # In[]
    windowed_stream = make_windows(data, labels_init, 2)
    win_over_thresh = variance_over_threshold(windowed_stream, 1, threshold=1.5e-6)
    bad_ones = []
    for x in win_over_thresh:
        if x[2]:
            bad_ones.append(x)
# In[]
    bad_ind = 35
    plt.plot(bad_ones[bad_ind][0])
    plt.show()

    # print(reduce(lambda x, init: init + int(x[2]), win_over_thresh), 0)
    ex_data = data[0]
    variance_first_file = get_variance(ex_data, 1)
    where_big_var = np.where(variance_first_file > 1.5e-6)
    # In[]


    plt.figure(figsize=(12, 6))
    plt.plot(variance_first_file)
    plt.show()
    # In[]
    ex_big_var_ind = np.random.choice(where_big_var[0])
    ex_good_sec = ex_data[10000:10500]
    ex_one_sec = ex_data[ex_big_var_ind - 250:ex_big_var_ind + 250]
    plt.plot(ex_one_sec)
    plt.title(f'apporx seconds: {ex_big_var_ind}: variance: {variance_first_file[ex_big_var_ind]}')
    plt.plot(ex_good_sec)
    plt.show()
# In[]


