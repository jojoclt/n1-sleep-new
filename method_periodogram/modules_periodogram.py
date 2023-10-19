from scipy.integrate import simps
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def get_periodogram_n1(signal_df, psg_id, sf=100, freq_low=4, freq_high=8):
    low, high = freq_low, freq_high

    win = 3 * sf
    N = len(signal_df)

    arr = []
    val = abs(signal_df).to_numpy()

    for i in range(0, N, 3000):
        max_len = min(i+3000, N)
        freqs, psd = signal.welch(val[i:max_len], sf, nperseg=win)
        idx = np.logical_and(freqs >= low, freqs <= high)
        freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25

        # Compute the absolute power by approximating the area under the curve
        power = simps(psd[idx], dx=freq_res)
        # print('Absolute ? power: %.3f uV^2' % power)
        arr.append(power)
    return arr
    # plt.plot(arr)
    # plt.title(f'? Power of {psg_id}')
    # plt.xlabel('Time')
    # plt.ylabel('? Power')
    # plt.show()
