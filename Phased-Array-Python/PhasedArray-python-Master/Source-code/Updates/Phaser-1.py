import adi
import scipy
import time
from scipy import signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import numpy as np

# Create instance
# sdr = adi.Pluto(uri="ip:192.168.2.1")
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
    element.rx_gain = 0x67
phaser.latch_rx_settings()


def Pluto_Rx(pluto_rx):
    # Setup Frequency and Test Tones
    pluto_rx.rx_lo = 4492000000
    freq_band = 400000
    tt1_freq = 000000
    tt2_freq = 000000
    buf_len = 2 ** 13
    # window = np.hamming(buf_len)
    avg_step = 0
    fft_rxvals_iq = np.array([])
    fftfreq_rxvals_iq = np.array([])
    avgband = np.array([])
    time.sleep(0.1)
    # adi.rx_tx.rx.rx_buffer_size=buf_len

    # Get and Plot Data
    for i in range(0, 50):
        pluto_rx.dds_frequencies = (str(tt1_freq), str(tt2_freq), str(tt1_freq), str(tt2_freq))
        # for j in range(5):
        data = pluto_rx.rx()
        window = np.hamming(len(data))
        data_r = window * data.real
        data_i = window * data.imag
        data_complex = data_r + (1j * data_i)
        avgband = np.abs(np.fft.fft(data_complex))
        freq = (np.fft.fftfreq(int(buf_len), 1 / int(pluto_rx.rx_lo)) + (pluto_rx.rx_lo + int(freq_band) * i)) / 1e6
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
        # plt.xlim(4500, 5500)
        # plt.ylim(-90, -10)
        plt.plot(fftfreq_rxvals_iq, fft_rxvals_iq_db)
        plt.draw()
        plt.pause(0.05)
        time.sleep(0.05)
        avg_step += 1

    # Uncomment the following section if you want to reset variables for continous plot

    # window = np.hamming(buf_len)
    # avg_step = 0
    # fft_rxvals_iq = np.array([])
    # fftfreq_rxvals_iq = np.array([])
    # avgband = np.array([])
    # time.sleep(0.1)
    # print("Done")