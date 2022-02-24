""" This Demo can be used to showcase beamwidth vs Number of elements as well as beamwidth tapering experiment
    For the beamwidth vs Number of element experiment 1st give gain [127, 127, 127, 127, 127, 127, 127, 127]. Now
    re-run the demo set gain of outer two elements 0 i.e. [0, 127, 127, 127, 127, 127, 127, 0] to see increase in width
    of beam. Continue this and repeat for outer 4 elements 0.
    For beam Tapering demo set the gain of middle element to 100%, adjacent elements to 75% then 50% and then 25%
    i.e. [32, 63, 95, 127, 127, 95, 63, 32]."""

import matplotlib.pyplot as plt
import time
from CN0566 import sdr, pll, beamformer

rpi_ip = "ip:analog.local"  # IP address of the Raspberry Pi
lo_ip = "ip:192.168.2.1"  # IP address of the Transreceiver Block

# Instantiate all the Devices
my_pll = pll(uri=rpi_ip)
my_sdr = sdr(uri=lo_ip)
my_beamformer = beamformer(
    uri=rpi_ip,
    chip_ids=["BEAM0", "BEAM1"],
    device_map=[[1], [2]],
    element_map=[[1, 2, 3, 4], [5, 6, 7, 8]],
    device_element_map={
        1: [2, 1, 4, 3],
        2: [6, 5, 8, 7],
    },
)

my_sdr.configure()
my_pll.configure()
my_beamformer.configure(device_mode="rx")
my_beamformer.rx_dev = my_sdr

my_beamformer.load_gain_cal('gain_cal_val.pkl')
my_beamformer.load_phase_cal('phase_cal_val.pkl')

gain_list = []
""" Note Channel 1 has index 0 so chan 0 corresponds to element 1."""
for i in range(0, 8):
    val = int(input(f"Enter gain of element {(i + 1)}\n"))
    gain_list.append(val)

print(gain_list)

for i in range(0, len(gain_list)):
    my_beamformer.set_chan_gain(i, gain_list[i])
    # print(i, gain_list[i])
x, y = my_beamformer.plot(plot_type="monopulse")
plt.clf()
plt.scatter(y, x)
plt.show()

