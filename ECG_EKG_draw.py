import os
import mne
import matplotlib.pyplot as plt
from MyEDFImports import import_ecg, get_edf_filenames, import_eeg

efds = get_edf_filenames()
tmin, tmax = 100, 101
ecg = import_ecg(name=efds[0], tmin=tmin, tmax=tmax)
one_eeg = import_eeg(name=efds[0], tmin=tmin, tmax=tmax)

# printing info
print(one_eeg.info)
print(ecg.info)
# plotting ECG
y = ecg[0][0].T
x = ecg[0][1]
plt.figure(figsize=(20, 6))
plt.step(x, y, )
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.legend(['ECG'])
plt.xlabel('time [s]')
plt.ylabel('voltage [V]')
plt.title(f"ECG of CN223100.edf from {tmin}s to {tmax}s")
plt.show()
# Plotting EEG
y = one_eeg[0][0].T
x = one_eeg[0][1]
plt.figure(figsize=(20, 6))
plt.step(x, y)
ax = plt.gca()
ax.invert_yaxis()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.legend(['EEG'])
plt.xlabel('time [s]')
plt.ylabel('voltage [V]')
plt.title(f"FP1A2 EEG of CN223100.edf from {tmin}s to {tmax}s")
plt.show()
