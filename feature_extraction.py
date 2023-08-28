import features as f
import MyEDFImports as m
import numpy as np
from hrvanalysis import get_time_domain_features

if __name__ == '__main__':
    func_names = list(f.get_all_functions_dict().keys())
    func_names.remove('RMSSD')
    print(func_names)

    ########################### DATA PROCESSING ###################################
    filename = 'r_peaks_220s'
    peaks = m.load_r_peaks_csv(filename)
    # for shallow sleep
    example_peaks = peaks[1]
    rrs = [np.diff(p) for p in example_peaks]
    one_window_rrs = example_peaks[100]
    hpvs = f.compute_hrv_metrics(one_window_rrs)
    print(hpvs)
    # In[]
    features = f.feature_extract(window_stream=rrs, features=func_names)