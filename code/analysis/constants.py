import numpy as np
from scipy.stats import norm

SAMPLE = False
width, height = 320, 240
total_mice=9
start_date = "20211125"

colours = ["#E33237", "#1C9E77", "F98E23", "#2E3192"]

significance_threshold = 0.05

camera_n = 6
cameras = np.arange(camera_n) + 1

dlc_suffix = "DLC_resnet_50_AnesthesiaDec9shuffle1_1000000_filtered.h5"
cutoff = .6
offset = 8 #seconds


def find_nearest(array, value):
    return (np.abs(array - value)).argmin()

def moving_average(arr, N = 3) :
    """
    a = numpy.ndarray
    N = window
    """
    arr_padded = np.pad(arr, (N//2, N-1-N//2), mode='edge')
    return np.convolve(arr_padded, np.ones((N,))/N, mode='valid')


def cutoff_condition(arr, cutoff):
    return (arr < cutoff), lambda z: z.nonzero()[0]

def to_rgba(hex):
    hex = hex.lstrip('#')
    tpl = tuple(int(hex[i:i + 2], 16)/255 for i in (0, 2, 4))
    return (*tpl, 1)

rgb_colours = [to_rgba(c) for c in colours]


def permutation_test(d1, d2, plot = False, samples = 50000):
    print(f"PERMUTATION TEST, samples: {samples}")
    diff_real = np.mean(d1) - np.mean(d2)
    d1_len = len(d1)
    d2_len = len(d2)
    combined = np.hstack((d1,d2)).flatten()
    idx = np.arange(combined.shape[0])
    diffs_shuffle = np.zeros(samples, dtype=np.float64)

    for sample in np.arange(samples):
        mask = np.ones(combined.shape[0], dtype=bool)
        d1_idx = np.random.choice(idx, d1_len, replace = False)

        mask[d1_idx] = False
        d2_idx = idx[mask]
        d1_shuffle = combined[d1_idx]
        d2_shuffle = combined[d2_idx]
        diffs_shuffle[sample] = np.mean(d1_shuffle) - np.mean(d2_shuffle)

    p_value = (diffs_shuffle <= diff_real).sum()/samples
    print(f"P-value: {p_value}\n# {('SIGNIFICANT' if p_value < significance_threshold else 'NON-SIGNIFICANT')} #")
    ax=None
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize =(3, 3))
        fig.canvas.set_window_title('Permutation test')
        ax.hist(diffs_shuffle, bins = samples//100, density=True, color="grey", alpha=.5)

        xmin, xmax = ax.get_xlim()

        ax.axvline(diff_real, color=rgb_colours[0],lw=1,linestyle="--")

        ax.text(diff_real -2, .01, f'P~{round(p_value, 2)}', style='italic', font="arial", fontsize=8, ha ="right", in_layout=True, color="k")

        #add normal dist
        mu, std = norm.fit(diffs_shuffle)
        x = np.linspace(xmin, xmax, 300)
        pdf = norm.pdf(x, mu, std)
        ax.plot(x, pdf, color=rgb_colours[-1],lw=1)
        crop=np.argmax(x>=diff_real)
        ax.plot(x[:crop], pdf[:crop], color=rgb_colours[0],lw=1)
        ax.fill_between(x[:crop], pdf[:crop], color = rgb_colours[0], alpha=.5)

        ax.locator_params(axis = 'y', nbins = 6)

        ax.set_ylabel("Probability density")
        ax.set_xlabel("Difference in mean")
        ax.set_yticks([0,0.03,0.06])
        ax.set_yticklabels(["0%","3%","6%"])

        fig.tight_layout()


    return p_value, ax

def reject_outliers(data, fdata, m = 2):
    data = data.copy()
    #print(data.shape)
    mean=np.median(fdata, axis = 0)
    data[abs(data - mean) > m * np.std(fdata, axis = 0)] = np.nan#mean[abs(data - mean) > m * np.std(fdata, axis = 0)]
    return data
    data = data.copy()
    d = np.abs(data - np.median(fdata, axis = 0))

    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    data[s > m] = np.nan

    return data
