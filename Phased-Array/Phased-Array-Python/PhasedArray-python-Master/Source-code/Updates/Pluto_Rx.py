import adi
import numpy as np
import matplotlib.pyplot as plt
import time

pluto_Rx = adi.Pluto('ip:10.84.10.64')
pluto_Rx.sample_rate = 3000 * 10**3
# pluto_Rx.tx_lo = 3000 * 10**6
pluto_Rx.rx_lo = 4000 * 10**6
pluto_Rx.rx_rf_bandwidth = 50 * 10**6
pluto_Rx.rx_hardwaregain_chan0 = -80

x = pluto_Rx.rx_lo


for r in range(20):
    data = pluto_Rx.rx()
    N = len(data)
    win = np.hamming(N)
    data_r = win*data.real
    data_i = win*data.imag
    data_complex = data_r + (1j * data_i)
    amp = np.abs(np.fft.fft(data_complex))
    amp_db = 20* np.log10(amp)
    # sig = np.fft.fft(data_complex)
    # plt.xlim([x-(5*(10**6)), x+(5*(10**6))])
    plt.clf()
    plt.plot(amp_db)
    plt.draw()
    plt.pause(0.05)
    time.sleep(0.05)
plt.show()