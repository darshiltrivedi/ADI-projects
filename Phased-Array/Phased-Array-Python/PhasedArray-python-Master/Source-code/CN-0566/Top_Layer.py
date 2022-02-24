import matplotlib.pyplot as plt
import time
""" This is the CN0566 directory and files created inside it. For now sdr pll and beamformer class are inside 
    3 different python files and CN0566 directory has it's own __init__.py file. Later merge them to single or change
    according to pyadi requirements."""
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
""" Configure all the device as required in Phased array beamformer project
    Current freq plan is Sig Freq = 10.492 GHz, antenna element spacing = 0.015in Freq of pll is 12/2 GHz
    this is mixed down using mixer to get 10.492 - 6 = 4.492GHz which is freq of sdr.
    This frequency plan can be updated any time in example code
    e.g:- my_beamformer.SigFreq = 9Ghz etc"""
my_sdr.configure()  # configure sdr/pluto according to above-mentioned freq plan
my_pll.configure()  # configure pll/adf4159 according to above-mentioned freq plan
# By default device_mode is "rx"
my_beamformer.configure(device_mode="rx")  # Configure adar in mentioned mode and also sets gain of all channel to 127

""" beamformer/adar1000 uses sdr to plot the output. In order to solve that dependency we can instantiate the 
    sdr device in beamformer class but that requires pass extra argument i.e. ip of sdr in beamformer as SDR 
    can be connected to raspberry pi which has all other devices or a different PC that can have diff ip.
    Alternate option is specify sdr/rx_dev instantiated on top layer like done here"""
my_beamformer.rx_dev = my_sdr

""" This can be useful in Array size vs beam width experiment or beamtappering experiment. 
    Set the gain of outer channels to 0 and beam width will increase and so on."""
my_beamformer.set_chan_gain(3, 120)  # set gain of Individual channel
my_beamformer.set_all_gain(120)  # Set all gain to mentioned value, if not, set to 127 i.e. max gain

""" To set gain of all channels with different values."""
gain_list = [0, 127, 127, 127, 127, 127, 127, 0]
for i in range(0, len(gain_list)):
    my_beamformer.set_chan_gain(i, gain_list[i])

""" If you want to use previously calibrated values load_gain and load_phase values by passing path of previously
    stored values. If this is not done system will be working as uncalibrated system."""
my_beamformer.load_gain_cal('gain_cal_val.pkl')
my_beamformer.load_phase_cal('phase_cal_val.pkl')

""" Averages decide number of time samples are taken to plot and/or calibrate system. By default it is 1."""
my_beamformer.Averages = 5

""" This instantiate calibration routine and perform gain and phase calibration. Note gain calibration should be always
    done 1st as phase calibration depends on gain cal values if not it throws error"""
my_beamformer.gain_calibration()   # Start Gain Calibration
my_beamformer.phase_calibration()  # Start Phase Calibration

""" This can be used to change the angle of center lobe i.e if we want to concentrate main lobe/beam at 45 degress"""
my_beamformer.set_beam_angle(45)

""" Plot the output based on experiment that you are performing"""
while True:
    x, y = my_beamformer.plot(plot_type="monopulse")
    plt.clf()
    plt.scatter(y, x)
    plt.draw()
    plt.pause(0.05)
    time.sleep(0.05)
