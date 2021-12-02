import adi
import numpy as np
import matplotlib.pyplot as plt
import time

pluto_tx = adi.Pluto(uri="ip:192.168.2.1")  # Pluto on RPI doesn't work with pyadi

# Setup Frequency and Test Tones
freq_band = 400000
tt1_freq = 000000
tt2_freq = 000000
buf_len = 2**13
# center_freq = np.array([])

#Setup Pluto
pluto_tx._tx_buffer_size = 2**13
pluto_tx.rx_buffer_size = 2**13
pluto_tx.tx_rf_bandwidth = 18000000
pluto_tx.tx_lo = 3000000000
pluto_tx.rx_lo = 3000000000
pluto_tx.tx_cyclic_buffer = True
pluto_tx.tx_hardwaregain = -30
pluto_tx.rx_hardwaregain_chan0 = 80
pluto_tx.gain_control_mode = 'slow_attack'
pluto_tx.sample_rate = 3044000
pluto_tx.dds_scales = ('0.1', '0.25', '0','0' , '0.1', '0.25', '0','0')
pluto_tx.dds_enabled = (1, 1, 1, 1, 1, 1, 1, 1)
pluto_tx.filter="LTE20_MHz.ftr"

window = np.hamming(buf_len)
avg_step = 0
fft_rxvals_iq = np.array([])
fftfreq_rxvals_iq = np.array([])
avgband = np.array([])
time.sleep(0.1)

while True:
    for i in range(0, 1):
        pluto_tx.dds_frequencies=(str(tt1_freq),str(tt2_freq),str(tt1_freq),str(tt2_freq),0,0,0,0)
        data = pluto_tx.rx()
        data_r = window * data.real
        data_i = window * data.imag
        data_complex = data_r + (1j * data_i)
        avgband = np.abs(np.fft.fft(data_complex))
        freq = (np.fft.fftfreq(int(buf_len), 1/int(pluto_tx.rx_lo)) + (pluto_tx.rx_lo + int(freq_band) * i)) /1e6
        tuple1 = zip(freq, avgband)
        tuple1 = sorted(tuple1, key=lambda x: x[0])
        iq = np.array([nr[1] for nr in tuple1])

        if avg_step == 0:
            fft_rxvals_iq = np.append(fft_rxvals_iq, iq)
            fftfreq_rxvals_iq = np.append(fftfreq_rxvals_iq, [nr[0] for nr in tuple1])
        else:
            # replace older amplitudes per band with newer amplitudes
            iq = iq * 1 / avg_step + fft_rxvals_iq * (avg_step - 1) / avg_step
            fft_rxvals_iq = iq

        fft_rxvals_iq_db = 20 * np.log10(fft_rxvals_iq * 2 / (2 ** 11 * buf_len))
        plt.clf()
        # plt.xlim(2500,3500)
        # plt.ylim(-90,-10)
        plt.plot(fftfreq_rxvals_iq, fft_rxvals_iq_db)
        plt.draw()
        plt.pause(0.05)
        time.sleep(0.05)
        avg_step += 1
    print(fftfreq_rxvals_iq[np.argmax(fft_rxvals_iq_db)])
    window = np.hamming(buf_len)
    avg_step = 0
    fft_rxvals_iq = np.array([])
    fftfreq_rxvals_iq = np.array([])
    avgband = np.array([])
    time.sleep(0.1)
