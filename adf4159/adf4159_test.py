import adi
import time

pll = adi.adf4159(uri="ip:10.84.10.66")
# print(pll.ramp_mode)
pll.frequency = 5900000000 # Output frequency divided by 2
pll.enable = 0 # Power down
pll.freq_dev_range = 0 # frequency deviation range
pll.freq_dev_step = 5690 # frequency deviation step
pll.freq_dev_time = 0 # frequency deviation time

while True:  # probe vtune and check if the values change after each 5 seconds
    pll.ramp_mode = "disabled"
    print("disabled")
    time.sleep(5)
    pll.ramp_mode = "continuous_sawtooth"
    print("cst")
    time.sleep(5)
    pll.ramp_mode = "continuous_triangular"
    print("ctt")
    time.sleep(5)
    pll.ramp_mode = "single_sawtooth_burst"
    print("ssb")
    time.sleep(5)
    pll.ramp_mode = "single_ramp_burst"
    print("srb")
    time.sleep(5)
    print("Done")
