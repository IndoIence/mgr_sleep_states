import csv
import os

import matplotlib.pyplot as plt
import neurokit2 as nk2
import numpy as np

import MyEDFImports as m

s_freq = 500


def load_data_labels(names):
    raw_data_all = []
    labels_all = []
    for edf_filename in names:
        edf_labels = m.load_labels_one_file(edf_filename)
        raw = m.import_ecg(edf_filename)
        data = raw[0][0].T
        raw_data_all.append(data)
        labels_all.append(edf_labels)

    return raw_data_all, labels_all


def make_windows(data_list, labels_list, win_pad):
    assert len(data_list) == len(labels_list)
    assert win_pad > 0 and type(win_pad) == int
    win_len = (2 * win_pad + 1) * 500 * 20
    twenty_sec = 20 * 500
    for sleep_data, labels in zip(data_list, labels_list):
        n_wide_wind = len(labels) - win_pad * 2
        # cutting last few unlabeled seconds
        labeled_data = sleep_data[:len(labels) * len(labels)]
        middle_labels = labels[win_pad:n_wide_wind + win_pad]
        for i, label in enumerate(middle_labels):
            yield labeled_data[i * twenty_sec:(i * twenty_sec) + win_len], label


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
        print(f"Eror deleting in {dir}")


def save_to_csv(windows_r_peaks_stream):
    csv_writers = {}
    open_files = {}
    # i should append for every new peak_list, but delete at first the file
    for win, peaks, label in windows_r_peaks_stream:
        if label == 5:
            print(peaks, len(win))
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


if __name__ == '__main__':
    edf_filenames = m.get_edf_filenames()[:1]
    # get data for every patient in a list of np arrays
    data, labels_init = load_data_labels(edf_filenames)
    # TODO: investigate why I am getting 0 files with label 5
    # In[]
    windows_stream = make_windows(data, labels_init, win_pad=2)
    windows, labels = zip(*windows_stream)
    lengths = [len(w) for w in windows]
    plt.plot(lengths)
    plt.show()
    #windows_r_peaks_stream = calc_r_peaks(windows_stream)
    #del_from_dir('r_peaks')
    #save_to_csv(windows_r_peaks_stream)

