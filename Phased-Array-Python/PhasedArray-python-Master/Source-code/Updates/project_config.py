#  Set all the configuration parameters in this File
#  Mention numbe of adars used transreceiver source all Constatnst and variables

sw_tx = 1  # Switch to toggle between Pluto and ad9361. Pluto == 1(1r1t) and ad9361 == 2(2r2t)
lo_ip = "ip:192.168.2.1"  # IP address of the Transreceiver Block
device_mode = "rx"  # Mode of operation of beamformer. It can be Tx, Rx or Detector

SignalFreq = 10492000000  # Frequency of source
Averages = 1  # Number of Avg too be taken. We be user input value on GUI
phase_step_size = 2.8125  # it is 360/2**number of bits. For now number of bits == 6. change later
steer_res = 2.8125  # It is steering resolution. This would be user selected value
c = 299792458  # speed of light in m/s
d = 0.015  # element to element spacing of the antenna
num_ADARs = 1  # address According to P10 jumper of EVAL-board
res_bits = 1  # res_bits and bits are two different vaiable. It can be variable but it is hardset to 1 for now

rx_gain = [0x7F, 0x7F, 0x7F, 0x7F]  # gain of individual channel
# Phase Calibration values
Rx1_Phase_Cal = 0
Rx2_Phase_Cal = 0
Rx3_Phase_Cal = 0
Rx4_Phase_Cal = 0
Rx5_Phase_Cal = 0
Rx6_Phase_Cal = 0
Rx7_Phase_Cal = 0
Rx8_Phase_Cal = 0
