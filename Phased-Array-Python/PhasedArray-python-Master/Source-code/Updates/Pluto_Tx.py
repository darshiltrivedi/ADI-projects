import adi
import numpy as np
import sys
from scipy import signal
import matplotlib.pyplot as plt
import time

# sys.path.append(r"C:\Users\DTrivedi\OneDrive - Analog Devices, Inc\Desktop\Work_database\Rotations\System Developement Group\Phased_Array\Pyadi-iio")
#Connect Pluto
sys.path.append('C:/apop/Git_Repositories/pyadi-iio')

pluto_tx = adi.Pluto(uri="ip:192.168.2.2")  # Pluto on RPI doesn't work with pyadi
phaser = adi.adar1000_array(uri="ip:analog.local", chip_ids=["BEAM0", "BEAM1", "BEAM2", "BEAM3"],
                                device_map=[[1, 2], [3, 4]],
                                element_map=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                                device_element_map={
                                    1: [5, 6, 2, 1],
                                    2: [7, 8, 4, 3],
                                    3: [13, 14, 10, 9],
                                    4: [15, 16, 12, 11],
                                },
                                )

# Configure ADAR1000
phaser.initialize_devices()  # Always Intialize the device 1st as reset is performed at Initialization

# Set the array frequency to 12GHz and the element spacing to 12.5mm so that the array can be accurately steered
phaser.frequency = 12e9
phaser.element_spacing = 0.015

for device in phaser.devices.values():
    device.mode = "rx"
    device.lna_bias_out_enable = False
    for channel in device.channels:
        channel.rx_enable = True
phaser.steer_rx(azimuth=10, elevation=45)

# Set the element gains to 0x67
for element in phaser.elements.values():
    element.rx_gain = 0x7F
phaser.latch_rx_settings()

# print(sdr.rx())



# Setup Frequency and Test Tones
freq_band = 400000
tt1_freq = 000000
tt2_freq = 000000
buf_len = 2**13
# center_freq = np.array([])



#Setup Pluto
pluto_tx._tx_buffer_size = 2**13
pluto_tx.rx_buffer_size = 2**13
pluto_tx.tx_rf_bandwidth = 55000000
pluto_tx.tx_lo = 6000000000
pluto_tx.rx_lo = 4492000000
pluto_tx.tx_cyclic_buffer = True
pluto_tx.tx_hardwaregain = -30
# pluto_tx.rx_hardwaregain_chan0 = 80
pluto_tx.gain_control_mode = 'slow_attack'
pluto_tx.sample_rate = 3044000
pluto_tx.dds_scales = ('0.3', '0.25', '0','0')
pluto_tx.dds_enabled = (1, 1, 1, 1)
pluto_tx.filter="LTE20_MHz.ftr"

time.sleep(0.1)

#Get and Plot Data

# ylim = [0,0]
window = np.hamming(buf_len)
avg_step = 0
fft_rxvals_iq = np.array([])
fftfreq_rxvals_iq = np.array([])
avgband = np.array([])
time.sleep(0.1)
# adi.rx_tx.rx.rx_buffer_size=buf_len

# Configure ADAR1000
# phaser.all_rx_gains()

while True:
    for i in range(0, 50):
        pluto_tx.dds_frequencies=(str(tt1_freq),str(tt2_freq),str(tt1_freq),str(tt2_freq))
        # for j in range(5):
        data = pluto_tx.rx()
        # window = np.hamming(len(data))
        data_r = window * data.real
        data_i = window * data.imag
        data_complex = data_r + (1j * data_i)
        avgband = np.abs(np.fft.fft(data_complex))
        freq = (np.fft.fftfreq(int(buf_len), 1/int(pluto_tx.rx_lo)) + (pluto_tx.rx_lo + int(freq_band) * i)) /1e6
        tuple1 = zip(freq, avgband)
        tuple1 = sorted(tuple1, key=lambda x: x[0])
        iq = np.array([nr[1] for nr in tuple1])

        if (avg_step == 0):
            fft_rxvals_iq = np.append(fft_rxvals_iq, iq)
            fftfreq_rxvals_iq = np.append(fftfreq_rxvals_iq, [nr[0] for nr in tuple1])
        else:
            # replace older amplitudes per band with newer amplitudes
            iq = iq * 1 / avg_step + fft_rxvals_iq * (avg_step - 1) / avg_step
            fft_rxvals_iq = iq

        fft_rxvals_iq_db = 20 * np.log10(fft_rxvals_iq * 2 / (2 ** 11 * buf_len))
        plt.clf()
        plt.xlim(4500,5500)
        plt.ylim(-90,-10)
        plt.plot(fftfreq_rxvals_iq,fft_rxvals_iq_db)
        plt.draw()
        plt.pause(0.05)
        time.sleep(0.05)
        avg_step += 1
    window = np.hamming(buf_len)
    avg_step = 0
    fft_rxvals_iq = np.array([])
    fftfreq_rxvals_iq = np.array([])
    avgband = np.array([])
    time.sleep(0.1)
    print("Done")
    # plt.show()


# This code has no effect on Rx
# fs = pluto_tx.sample_rate
# fund_freq = pluto_tx.tx_lo
# t = np.arange(0, 1, 1/fs)
# amplitude = 2
# tx_sig = amplitude * np.sin(2 * np.pi * fund_freq * t)
# pluto_tx.tx(tx_sig)
