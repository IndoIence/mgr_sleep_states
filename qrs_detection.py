import sys

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
    # check if patient of this index exists
    if patient_nr > len(edf_filenames) or patient_nr < 0:
        print(f'there  are {len(edf_filenames)} patients')
        for i, filename in enumerate(edf_filenames):
            print(f'{i + 1}: {filename}')
        quit()
    # load data
    chosen_filename = edf_filenames[patient_nr]
    y = m.load_data_one_file(chosen_filename)
    if patient_nr == 1:
        y = y*-1
    window_len = y.shape[1] + 1
    x_one_window = np.linspace(0, window_len / sampling_freq, window_len, endpoint=False)
    labels = m.load_labels_one_file(chosen_filename)

    # check if given index exists
    if given_ind > len(labels) or given_ind < 0:
        print(f'chosen window: {given_ind} when there are {len(labels)} windows')
        quit()
    chosen_window = y[given_ind]
    detectors = Detectors(sampling_freq)
    peaks = detectors.engzee_detector(chosen_window)
    print(peaks)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=x_one_window,
        y=chosen_window,
        mode='lines'
    ))
    fig.add_trace(go.Scattergl(
        x=np.array(peaks) / sampling_freq,
        y=chosen_window[peaks],
        mode='markers'
    ))
    print('showing figure')
    fig.show()


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
