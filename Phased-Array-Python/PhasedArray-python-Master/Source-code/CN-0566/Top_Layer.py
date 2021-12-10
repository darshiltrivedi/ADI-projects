from cn0566 import *
from project_config import *

my_pll = pll.adf4159(uri=rpi_ip)
my_sdr = sdr(uri=lo_ip)
my_antenna_array = beamformer(
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
my_antenna_array.configure(device_mode="rx")
my_antenna_array.rx_dev = my_sdr
# my_antenna_array.set_chan_gain(3, 120)
# my_antenna_array.load_gain('gain_cal_val.pkl')
# my_antenna_array.set_all_gain()
# my_antenna_array.calculate_plot()
# my_antenna_array.set_beam_angle(45)
my_antenna_array.gain_calibration()
my_antenna_array.phase_calibration()
