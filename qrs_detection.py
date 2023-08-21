import sys

import matplotlib.pyplot as plt
import neurokit2
import wfdb.processing
import plotly.express as px
import MyEDFImports as m
import plotly.io as pio
import plotly.graph_objs as go
import numpy as np
from ecgdetectors import Detectors

sampling_freq = 500
if __name__ == "__main__":
    pio.renderers.default = "browser"
    edf_filenames = m.get_edf_filenames()
    # get arguments
    patient_nr = int(sys.argv[1])
    given_ind = int(sys.argv[2])
    # if there is 3rd arg get its det method
    det_method = int(sys.argv[3]) if len(sys.argv) == 4 else 2
    if det_method > 3:
        det_method = 3
    elif det_method < 1:
        det_method = 2
    # check if patient of this index exists
    if patient_nr > len(edf_filenames) or patient_nr < 0:
        print(f'there  are {len(edf_filenames)} patients')
        for i, filename in enumerate(edf_filenames):
            print(f'{i}: {filename}')
        quit()
    # load data
    chosen_filename = edf_filenames[patient_nr]
    y = m.load_data_one_file(chosen_filename)
    # if patient_nr == 1:
    #     y = y*-1
    window_len_plus_1 = y.shape[1] + 1
    x_one_window = np.linspace(0, window_len_plus_1 / sampling_freq, window_len_plus_1, endpoint=False)
    labels = m.load_labels_one_file(chosen_filename)

    # check if given index exists
    if given_ind > len(labels) or given_ind < 0:
        print(f'chosen window: {given_ind} when there are {len(labels)} windows')
        quit()
    chosen_window = y[given_ind]
    label_name = m.stages_names[labels[given_ind]]
    # three methods of peak detection
    peaks = [-1]
    if det_method == 1:
        detectors = Detectors(sampling_freq)
        peaks = detectors.engzee_detector(chosen_window)
    elif det_method == 2:
        _, results = neurokit2.ecg_peaks(chosen_window, sampling_rate=sampling_freq)
        peaks = results["ECG_R_Peaks"]
    elif det_method == 3:
        # debug
        peaks_entire_sleep = wfdb.processing.xqrs_detect(y.flatten(), fs=sampling_freq)
        start_filter, end_filter = given_ind*y.shape[1], (given_ind+1)*y.shape[1]
        peaks = np.array(list(filter(lambda x: start_filter <= x < end_filter, peaks_entire_sleep)))
        peaks = peaks - start_filter
    det_names = ['py-ecg-detectors', 'neurokit2', 'wfdb']

    print(peaks)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=x_one_window,
        y=chosen_window,
        mode='lines',
        name='ECG'
    ))
    fig.add_trace(go.Scattergl(
        x=np.array(peaks) / sampling_freq,
        y=chosen_window[peaks],
        mode='markers',
        name='R peaks'
    ))
    fig.update_layout(
        title=f'{edf_filenames[patient_nr]}: {given_ind}: label: {label_name}: detector: {det_names[det_method-1]}'
    )
# In[]
    fig.show()
# In[]
    import matplotlib.pyplot as plt
    print(np.diff(peaks) / 500)
    plt.plot(np.diff(peaks) / 500)
    plt.show()

#
#
# example_filename = edf_filenames[2]
# ex_edf = m.import_ecg(example_filename)
# y = ex_edf[0][0]
# x = ex_edf[0][1]
# # In[]
# y = m.load_data_one_file(example_filename)
# y = y.flatten()
# x = np.linspace(0, len(y + 1) / sampling_freq, len(y), endpoint=False)
# # In[]
#
# # In[1]
# fig = go.Figure()
# fig.add_trace(go.Scattergl(
#     x=x,
#     y=y,
#     mode='lines'
# ))
# fig.show()
# # In[3]
# N = 100
