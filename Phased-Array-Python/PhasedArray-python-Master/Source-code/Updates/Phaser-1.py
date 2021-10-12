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

# print(sdr.rx())

# # Configure Pluto
#
# sdr.gain_control_mode_chan0 = "manual"  # Receive path AGC Options: slow_attack, fast_attack, manual
# sdr.gain_control_mode_chan1 = "manual"
# sdr._ctrl.debug_attrs["adi,frequency-division-duplex-mode-enable"].value = "1"
# sdr._ctrl.debug_attrs["adi,ensm-enable-txnrx-control-enable"].value = "0"
# sdr._ctrl.debug_attrs["initialize"].value = "1"
# sdr._rxadc.set_kernel_buffers_count(1)
# rx = sdr._ctrl.find_channel('voltage0')
# # rx = sdr._ctrl.find_channel('voltage1')
# rx.attrs['quadrature_tracking_en'].value = '0'   # set to '1' to enable quadrature tracking
# sdr.rx_enabled_channels = [0,1] # Since I have Rev. C only 1 channel is working
# sdr.sample_rate = 3000000
# sdr.gain_control_mode_chan0 = 'manual'      #We must be in manual gain control mode (otherwise we won't see the peaks and nulls!)
# sdr.gain_control_mode_chan1 = 'manual'      #We must be in manual gain control mode (otherwise we won't see the peaks and nulls!)
# sdr.rx_buffer_size = int(1024)    # We only need a few samples to get the gain.  And a small buffer will greatly speed up the sweep
# sdr.tx_hardwaregain_chan0 = -80   # Make sure the Tx channels are attenuated (or off) and their freq is far away from Rx
# sdr.tx_hardwaregain_chan1 = -80
# # sdr.tx_lo = int(1000000000)
#
#
# # sdr.rx_rf_bandwidth = 55000000
# sdr.rx_lo = 4492000000
# # sdr.rx_hardwaregain_chan0 = 71  # dB,Gain applied to RX path.Only applicable when gain_control_mode is set to 'manual'
#
# sdr.tx_lo = int(6000000000)  # MHz, Carrier frequency of TX path
# # sdr.tx_hardwaregain_chan0 = -10  # dB, Attenuation applied to TX path
# # sdr.tx_rf_bandwidth = 55000000  # MHz, Bandwidth of front-end analog filter of TX path
# # sdr.tx_cyclic_buffer = True  # Toggles cyclic buffer
# # sdr.sample_rate = 30000000  # MSPS, Sample rate RX and TX paths
# # sdr.loopback = 0  # 0=Disabled, 1=Digital, 2=RF
# # sdr.filter =  "1.csv" #open("1.csv") #FIR filter
# # sdr.dds_single_tone(frequency=9279047, scale= 0)
#
#
# fs = sdr.sample_rate
# fund_freq = sdr.tx_lo
# t = np.arange(0, 1, 1/fs)
# amplitude = 1
# # samples = np.linspace(0, t, int(fs*t))
# tx_sig = amplitude * np.sin(2 * np.pi * fund_freq * t)
# # plt.plot(tx_sig)
# # plt.show()
#
# # Send data
#
# # Reading Pluto Data
# # total_sum=0
# # total_angle=0
# for i in range(50):
#     sdr.tx(tx_sig)
#     data = sdr.rx()
#     # N = len(data)
#     # win = np.hamming(N)
#     # y = win*data
#     # s_sum = np.fft.fftshift(y)
#     # max_index = np.argmax(s_sum)
#     # total_angle = total_angle + (np.angle(s_sum[max_index]))
#     # s_mag_sum = np.abs(s_sum[max_index])
#     # s_dbfs_sum = 20 * np.log10(np.max([s_mag_sum, 10 ** (-15)]) / (2 ** 12))
#     # total_sum = total_sum + (s_dbfs_sum)
#     # plt.plot(s_dbfs_sum)
#     # f, Pxx_den = signal.periodogram(data, fs)
#     z = fft(data, 16384)
#     z1 = 20 * np.log10(z)
#     plt.clf()
#     # plt.semilogy(f, Pxx_den)
#     plt.plot(z1)
#     plt.draw()
#     plt.pause(0.05)
#     time.sleep(0.05)
