import os
import matplotlib.pyplot as plt
import numpy as np
import mne

edf_dir = r"Jean-Pol_repaired_headers"


def get_edf_filenames(path=edf_dir):
    all_files = os.listdir(path)
    edf_files = list(filter(lambda x: x[-4:] == '.edf', all_files))
    return edf_files


def read_raw_edf(edf_path):
    e = mne.io.read_raw_edf(edf_path)
    return e


edfs = get_edf_filenames()
print(edfs)
print(edf_dir + "\\" + edfs[3])
edf = read_raw_edf(edf_dir + "\\" + edfs[0])

# croping the time, so the analysis is faster
edf.crop(tmin=60, tmax=120)
print("printing info:", edf)
print("Bad channels:", edf.info['bads'])
print("Sampling frequency:", edf.info['sfreq'], 'Hz')
print(edf.time_as_index(20))
print(edf.time_as_index([20,30,40]))
print(np.diff(edf.time_as_index([1,2,3])))

eeg_only = edf.copy().pick_types(eeg=True)
print("in my case all types are eeg")
print(len(edf.ch_names),len(eeg_only.ch_names))

y = edf[0][0].T
x = edf[0][1]

plt.plot(x,y)
plt.show()

chosen_ch_names = ['FZA2  EEG', 'F8A1  EEG']
two_ch = edf[chosen_ch_names]
y_offset = np.array([5e-4, 0])
x = two_ch[1]
y = two_ch[0].T + y_offset
lines = plt.plot(x,y)
plt.legend(lines, chosen_ch_names)
plt.show()

edf.plot(duration=5, n_channels=20)

edf.plot(duration=1, n_channels=5)
