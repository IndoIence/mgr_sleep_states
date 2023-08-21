import matplotlib.pyplot as plt
import numpy as np


def wplot(x, y=None, size=0, show=False, ax_equal=False, complex=0, **kw):
    if y is None:
        if complex:
            plt.plot(x.real, **kw)
            plt.plot(x.imag, **kw)
        else:
            plt.plot(x, **kw)
    else:
        if complex:
            plt.plot(x, y.real, **kw)
            plt.plot(x, y.imag, **kw)
        else:
            plt.plot(x, y.real, **kw)
    _scale_plot(plt.gcf(), plt.gca(), size=size, show=show, ax_equal=ax_equal)

def wscat(x, y=None, size=0, show=False, ax_equal=False, s=18, **kw):
    if y is None:
        plt.scatter(np.arange(len(x)), x, s=s, **kw)
    else:
        plt.scatter(x, y, s=s, **kw)
    _scale_plot(plt.gcf(), plt.gca(), size=size, show=show, ax_equal=ax_equal)

def mag_zoom(r, h=10, x0=0, show=True):
    N = len(r)
    N2 = N // 2
    a, b = N2 + x0 - h, N2 + x0 + h + 1
    k = np.linspace(-N2, N2 - 1, N)[a:b]
    plt.plot(k, r[a:b])
    _scale_plot(plt.gcf(), plt.gca(), show=show)


def imshow(data, norm=None, complex=None, abs=0, show=1, w=1, h=1,
           ridge=0, **kw):
    kw['interpolation'] = kw.get('interpolation', 'none')
    if norm is None:
        mx = np.max(np.abs(data))
        vmin, vmax = -mx, mx
    else:
        vmin, vmax = norm

    if abs:
        plt.imshow(np.abs(data), cmap='bone', vmin=0, vmax=vmax, **kw)
    else:
        if (complex is None and np.sum(np.abs(np.imag(data))) < 1e-8) or (
                complex is False):
            plt.imshow(np.real(data), cmap='bwr', vmin=vmin, vmax=vmax, **kw)
        else:
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(data.real, cmap='bwr', vmin=vmin, vmax=vmax, **kw)
            axes[1].imshow(data.imag, cmap='bwr', vmin=vmin, vmax=vmax, **kw)
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1,
                                wspace=0, hspace=0)
    plt.gcf().set_size_inches(14 * w, 8 * h)

    if ridge:
        data_mx = np.where(np.abs(data) == np.abs(data).max(axis=0))
        plt.scatter(data_mx[1], data_mx[0], color='r', s=4)
    if show:
        plt.show()


def _scale_plot(fig, ax, size=True, show=False, ax_equal=False):
    xmin, xmax = ax.get_xlim()
    rng = xmax - xmin
    ax.set_xlim(xmin + .018 * rng, xmax - .018 * rng)
    if size:
        fig.set_size_inches(15, 7)
    if ax_equal:
        yabsmax = max(np.abs([*ax.get_ylim()]))
        mx = max(yabsmax, max(np.abs([xmin, xmax])))
        ax.set_xlim(-mx, mx)
        ax.set_ylim(-mx, mx)
        fig.set_size_inches(8, 8)
    if show:
        plt.show()

def plotenergy(x, axis=1, **kw):
    wplot(np.sum(np.abs(x) ** 2, axis=axis), **kw)