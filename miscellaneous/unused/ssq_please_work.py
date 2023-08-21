"""Run locally, with ssqueezepy cloned"""
import numpy as np
import scipy.signal as sig
import pywt

from utils import wplot, wscat, imshow, allfft, plotenergy
from ssqueezepy.wavelet_transforms import cwt_fwd
from ssqueezepy.utils import p2up

#%%##########################################################################
def _scales(len_x, nv=32):
    N = p2up(len_x)[0]
    noct = np.log2(N) - 1
    n_scales = int(noct * nv)
    return np.power(2 ** (1 / nv), np.arange(1, n_scales + 1))

def _t(min, max, N):
    return np.linspace(min, max, N, False)

def cos_f(freqs, N=128):
    return np.concatenate([np.cos(2 * np.pi * f * _t(i, i + 1, N))
                           for i, f in enumerate(freqs)])
#%%##########################################################################
def test(x, l1_norm=1, _abs=1, ridge=1):
    scales = _scales(len(x))
    rscales = np.sqrt(scales).reshape(-1, 1)
    norm = rscales if l1_norm else 1
    coefp  = (pywt.cwt(x, scales, 'cmor2-0.796', method='conv', precision=14)[0]
              / (2 * np.pi)**(-.5))
    coefs  = sig.cwt(x, sig.morlet2, scales)
    coefss = cwt_fwd(x, 'morlet', opts={'padtype': 'symmetric'})[0]

    kw = dict(abs=_abs, complex=bool(_abs), ridge=ridge, w=.86, h=.9, aspect='auto')
    imshow(coefs / norm, **kw)
    imshow(coefp / norm, **kw)
    imshow(coefss/ norm, **kw)

    plotenergy((coefs / norm) / np.abs(coefs).max(),  axis=1)
    plotenergy((coefp / norm) / np.abs(coefp).max(),  axis=1)
    plotenergy((coefss/ norm) / np.abs(coefss).max(), axis=1)

    return coefs, coefp, coefss, rscales

#%%##########################################################################
N = 1024
t = np.linspace(0, 1, N, False)
x = (cos_f([8], N) + cos_f([32], N)
     + np.array([0] * 512 + list(cos_f([2], N)[:512]))
     + np.array([0] * 512 + list(cos_f([128], N)[:512]))
     )
coef, theta, r = allfft(x)

wplot(x); wscat(x, s=8, show=1)
wplot(np.abs(coef), show=1)
#%%###########################################################################
coefs, coefp, coefss, rscales = test(x, l1_norm=1, ridge=0)