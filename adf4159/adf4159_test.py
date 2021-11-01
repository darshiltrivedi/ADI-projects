import adi
import time

pll = adi.adf4159(uri="ip:10.84.10.66")
# print(pll.ramp_mode)
pll.frequency = 5900000000
pll.enable = 0
pll.freq_dev_range = 0
pll.freq_dev_step = 5690


# while True:
#     pll.ramp_mode = "disabled"
#     print("disabled")
#     time.sleep(5)
#     pll.ramp_mode = "continuous_sawtooth"
#     print("cst")
#     time.sleep(5)
#     pll.ramp_mode = "continuous_triangular"
#     print("ctt")
#     time.sleep(5)
#     pll.ramp_mode = "single_sawtooth_burst"
#     print("ssb")
#     time.sleep(5)
#     pll.ramp_mode = "single_ramp_burst"
#     print("srb")
#     time.sleep(5)
#     print("Done")
