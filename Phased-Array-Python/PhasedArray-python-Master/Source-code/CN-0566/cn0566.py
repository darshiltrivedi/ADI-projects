# Copyright (C) 2021 Analog Devices, Inc.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#     - Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     - Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the
#       distribution.
#     - Neither the name of Analog Devices, Inc. nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#     - The use of this software may or may not infringe the patent rights
#       of one or more patent holders.  This license does not release you
#       from the requirement that you obtain separate licenses from these
#       patent holders to use this software.
#     - Use of the software either in source or binary form, must be run
#       on or directly connected to an Analog Devices Inc. component.
#
# THIS SOFTWARE IS PROVIDED BY ANALOG DEVICES "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED.
#
# IN NO EVENT SHALL ANALOG DEVICES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, INTELLECTUAL PROPERTY
# RIGHTS, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from adi.adar1000 import adar1000, adar1000_array
from adi.adf4159 import adf4159
from adi.ad936x import ad9361, Pluto
import numpy as np
import pickle

''' 
    Phased Array beamformwer has 3 main devices 1. beamformer 2. Rx_device - SDR 3. Tx_device - pll
    All the 3 device's class are in the same file but spliting them may help to make it more clean
'''


class beamformer(adar1000_array):
    def __init__(self,
                 uri=None,
                 chip_ids=None,
                 device_map=None,
                 element_map=None,
                 device_element_map=None):

        """ Beamformer class inherits from adar1000_array and adds batch operations for phaser project
         like configure, calibrate set_beam_angle etc"""

        adar1000_array.__init__(self,
                                uri,
                                chip_ids,
                                device_map,
                                element_map,
                                device_element_map)

        """Initialize all the class variables for the project. Fixed variable allocation changed to dynamic """
        self.device_mode = None
        # self.Rx1_Phase_Cal, self.Rx2_Phase_Cal, self.Rx3_Phase_Cal, self.Rx4_Phase_Cal, self.Rx5_Phase_Cal,
        # self.Rx6_Phase_Cal, self.Rx7_Phase_Cal, self.Rx8_Phase_Cal = 0, 0, 0, 0, 0, 0, 0, 0
        self.SignalFreq = 10492000000  # Frequency of source
        self.Averages = 1  # Number of Avg too be taken. We be user input value on GUI
        self.phase_step_size = 2.8125  # it is 360/2**number of bits. For now number of bits == 6. change later
        self.steer_res = 2.8125  # It is steering resolution. This would be user selected value
        self.c = 299792458  # speed of light in m/s
        self.element_spacing = 0.015  # element to element spacing of the antenna
        self.res_bits = 1  # res_bits and bits are two different vaiable. It can be variable but it is hardset to 1 for now
        self.calibrated_values = []
        self.gcalibrated_values = []
        self.gcal = []  # [0x7F for i in range(0, 4*len(list(self.devices.values())))]
        self.rx_dev = None
        self.gain_cal = False
        self.phase_cal = False

    # This method configure the device beamformer props like RAM bypass, Transreciever source etc., based on device mode
    def configure(self, device_mode=""):
        self.device_mode = device_mode
        for device in self.devices.values():  # For device in Dict of device array
            # Configure ADAR1000
            # adar.initialize_devices()  # Always Intialize the device 1st as reset is performed at Initialization
            # If ADAR1000 array is used initialization work otherwise reset each adar individually
            device.reset()  # Performs a soft reset of the device (writes 0x81 to reg 0x00)
            device._ctrl.reg_write(0x400, 0x55)  # This trims the LDO value to approx. 1.8V (to the center of its range)
            device.sequencer_enable = False
            device.beam_mem_enable = False  # RAM control vs SPI control of the adar state, reg 0x38, bit 6.  False sets bit high and SPI control This code is writing value 40 to reg 38 Next line writes 20 so 60 in all. This is done by single command in SPI writes.
            device.bias_mem_enable = False  # RAM control vs SPI control of the bias state, reg 0x38, bit 5.  False sets bit high and SPI control
            device.pol_state = False  # Polarity switch state, reg 0x31, bit 0. True outputs -5V, False outputs 0V
            device.pol_switch_enable = False  # Enables switch driver for ADTR1107 switch, reg 0x31, bit 3
            device.tr_source = 'spi'  # TR source for chip, reg 0x31 bit 2.  'external' sets bit high, 'spi' sets bit low
            device.tr_spi = 'rx'  # TR SPI control, reg 0x31 bit 1.  'tx' sets bit high, 'rx' sets bit low
            device.tr_switch_enable = True  # Switch driver for external switch, reg0x31, bit 4
            device.external_tr_polarity = True  # Sets polarity of TR switch compared to TR state of ADAR1000.  True outputs 0V in Rx mode

            device.rx_vga_enable = True  # Enables Rx VGA, reg 0x2E, bit 0.
            device.rx_vm_enable = True  # Enables Rx VGA, reg 0x2E, bit 1.
            device.rx_lna_enable = True  # Enables Rx LNA, reg 0x2E, bit 2. bit3,4,5,6 enables RX for all the channels
            device._ctrl.reg_write(0x2E,
                                   0x7F)  # bit3,4,5,6 enables RX for all the channels. It is never set if not in this line
            device.rx_lna_bias_current = 8  # Sets the LNA bias to the middle of its range
            device.rx_vga_vm_bias_current = 22  # Sets the VGA and vector modulator bias.  I thought this should be 22d, but Stingray has it as 85d????

            device.tx_vga_enable = True  # Enables Tx VGA, reg 0x2F, bit0
            device.tx_vm_enable = True  # Enables Tx Vector Modulator, reg 0x2F, bit1
            device.tx_pa_enable = True  # Enables Tx channel drivers, reg 0x2F, bit2
            device.tx_pa_bias_current = 6  # Sets Tx driver bias current
            device.tx_vga_vm_bias_current = 22  # Sets Tx VGA and VM bias.  I thought this should be 22d, but stingray has as 45d??????

            if self.device_mode == "rx":
                # Configure the device for Rx mode
                device.mode = "rx"  # Mode of operation, bit 5 of reg 0x31. "rx", "tx", or "disabled".

                SELF_BIASED_LNAs = True
                if SELF_BIASED_LNAs:
                    # Allow the external LNAs to self-bias
                    device.lna_bias_out_enable = False  # this writes 0xA0 0x30 0x00.  What does this do? # Disabling it allows LNAs to stay in self bias mode all the time
                    # self._ctrl.reg_write(0x30, 0x00)   #Disables PA and DAC bias
                else:
                    # Set the external LNA bias
                    device.lna_bias_on = -0.7  # this writes 0x25 to register 0x2D.  This is correct.  But oddly enough, it doesn't first write 0x18 to reg 0x00....
                    # self._ctrl.reg_write(0x30, 0x20)   #Enables PA and DAC bias.  I think this would be needed too?

                # Enable the Rx path for each channel
                for channel in device.channels:
                    channel.rx_enable = True  # this writes reg0x2E with data 0x00, then reg0x2E with data 0x20.  So it overwrites 0x2E, and enables only one channel
                    # self._ctrl.reg_write(0x2E, 0x7F)    #Enables all four Rx channels, the Rx LNA, Rx Vector Modulator and Rx VGA

            # Configure the device for Tx mode
            elif self.device_mode == "tx":
                device.mode = "tx"

                # Enable the Tx path for each channel and set the external PA bias
                for channel in device.channels:
                    channel.tx_enable = True
                    channel.pa_bias_on = -2

            else:
                raise ValueError("Configure Device in proper mode")

            if self.device_mode == "rx":
                device.latch_rx_settings()  # writes 0x01 to reg 0x28.
            elif self.device_mode == "tx":
                device.latch_tx_settings()  # writes 0x02 to reg 0x28.

    # This method load gain calibrated value from filepath specified, if not calibrated set all channel gain to max
    def load_gain(self, filename=''):
        # I used pickle file to serialize the data. Maybe could switch to more standard format like JSON or csv
        try:
            with open(filename, 'rb') as file1:
                self.gcal = pickle.load(file1)
        except:
            for i in range(0, (4 * len(list(self.devices.values())))):
                self.gcal.append(0x7F)

    # This will try to set all the gain to it's calibrated values. If system is not calibrated set all chans to max gain
    def set_all_gain(self):
        self.load_gain()
        j = 0
        for device in self.devices.values():
            channel_gain_value = []
            for ind in range(0, 4):
                channel_gain_value.append(self.gcal[((j * 4) + ind)])
            j += 1
            i = 0  # Gain of Individual channel
            for channel in device.channels:
                if self.device_mode == 'rx':
                    channel.rx_gain = channel_gain_value[i]
                elif self.device_mode == 'tx':
                    channel.tx_gain = channel_gain_value[i]
                else:
                    raise ValueError("Configure the device first")
                i += 1
            if self.device_mode == 'rx':
                device.latch_rx_settings()  # writes 0x01 to reg 0x28
            elif self.device_mode == 'tx':
                device.latch_tx_settings()  # writes 0x01 to reg 0x28

    # This method load phase calibrated value from filepath specified, if not cali set all channel phase correction to 0
    def load_phase(self, filename=''):
        try:
            with open(filename, 'rb') as file:
                self.calibrated_values = pickle.load(file)
        except:
            for i in range(0, (4 * len(list(self.devices.values())))):
                self.calibrated_values.append(0)

    # This allows to set individual gain of the channel
    def set_chan_gain(self, chan_no, gain_val):  # chan_no is the ch whose gain you want to set, gain_val is value
        if self.device_mode == 'rx':
            list(self.devices.values())[chan_no // 4].channels[(chan_no - (4 * (chan_no // 4)))].rx_gain = gain_val
        elif self.device_mode == 'tx':
            list(self.devices.values())[chan_no // 4].channels[(chan_no - (4 * (chan_no // 4)))].tx_gain = gain_val
        else:
            raise ValueError("Configure the device first")

    """ A public method to sweep the phase value from -180 to 180 deg, calculate phase values of all the channel
      and set them. If we want beam angle at fixed angle you can pass angle value at which you want center lobe"""

    def set_beam_angle(self, Ph_Diff):
        self.load_phase('phase_cal_val.pkl')
        adar_list = list(self.devices.values())
        j = 0
        for adar in adar_list:
            channel_phase_value = []
            for ind in range(0, 4):
                channel_phase_value.append(((np.rint(Ph_Diff * ((j * 4) + ind) / self.phase_step_size) *
                                             self.phase_step_size) + self.calibrated_values[((j * 4) + ind)]) % 360)
            # print(channel_phase_value)
            j += 1
            i = 0
            for channel in adar.channels:
                # Set phase depending on the device mode
                if self.device_mode == "rx":
                    channel.rx_phase = channel_phase_value[
                        i]  # writes to I and Q registers values according to Table 13-16 from datasheet.
                i = i + 1
            if self.device_mode == "rx":
                adar.latch_rx_settings()
            else:
                adar.latch_tx_settings()

    """ This method is not complete yet. It plots a graph directly for now. This method calls set_beam_angle and 
        sets the Phases of all channel."""
    def __calculate_plot(self, gcal_element=0, cal_element=0):
        # self.load_phase('phase_cal_val.pkl')
        PhaseValues = np.arange(-196.875, 196.875,
                                self.phase_step_size)  # These are all the phase deltas (i.e. phase difference between Rx1 and Rx2, then Rx2 and Rx3, etc.) we'll sweep.
        PhaseStepNumber = 0  # this is the number of phase steps we'll take (140 in total).  At each phase step, we set the individual phases of each of the Rx channels
        max_signal = -100000  # Reset max_signal.  We'll keep track of the maximum signal we get as we do this 140 loop.
        max_angle = -90  # Reset max_angle.  This is the angle where we saw the max signal.  This is where our compass will point.
        gain = []
        delta = []
        beam_phase = []
        angle = []
        diff_error = []
        for PhDelta in PhaseValues:  # This sweeps phase value from -180 to 180
            if self.gain_cal:
                self.__set_gain_phase(PhDelta, gcal_element)
            if self.phase_cal:
                self.__set_phase_phase(PhDelta, cal_element)
            if not self.gain_cal and not self.phase_cal:
                self.set_beam_angle(PhDelta)
            value1 = (self.c * np.radians(np.abs(PhDelta))) / (2 * 3.14159 * self.SignalFreq * self.element_spacing)
            clamped_value1 = max(min(1, value1),
                                 -1)  # arcsin argument must be between 1 and -1, or numpy will throw a warning
            theta = np.degrees(np.arcsin(clamped_value1))
            if PhDelta >= 0:
                SteerAngle = theta  # positive PhaseDelta covers 0deg to 90 deg
            else:
                SteerAngle = -theta  # negative phase delta covers 0 deg to -90 deg

            total_sum = 0
            total_delta = 0
            total_angle = 0
            for count in range(0, self.Averages):  # repeat loop and average the results
                data = self.rx_dev.rx()  # read a buffer of data from Pluto using pyadi-iio library (adi.py)
                chan1 = data[
                    0]  # Rx1 data. data is a list of values chan1 and chan2 are just 1st and 2nd element of list Do we have to discard all other values?
                chan2 = data[1]  # Rx2 data. Changing data[0] to data. delta chan is all 0.
                sum_chan = chan1 + chan2
                delta_chan = chan1 - chan2
                N = len(sum_chan)  # number of samples  len(sum_chan) = 1 as just 1st element of list is taken
                win = np.blackman(N)
                y_sum = sum_chan * win
                y_delta = delta_chan * win

                sp = np.absolute(np.fft.fft(y_sum))
                sp = sp[1:-1]
                s_sum = np.fft.fftshift(sp)

                dp = np.absolute(np.fft.fft(y_delta))
                dp = dp[1:-1]
                s_delta = np.fft.fftshift(dp)

                max_index = np.argmax(s_sum)
                total_angle = total_angle + (np.angle(s_sum[max_index]) - np.angle(s_delta[max_index]))

                s_mag_sum = np.abs(s_sum[max_index]) * 2 / np.sum(win)
                s_mag_delta = np.abs(s_delta[max_index]) * 2 / np.sum(win)
                s_mag_sum = np.maximum(s_mag_sum, 10 ** (-15))
                s_mag_delta = np.maximum(s_mag_delta, 10 ** (-15))
                s_dbfs_sum = 20 * np.log10(
                    s_mag_sum / (2 ** 12))  # make sure the log10 argument isn't zero (hence np.max)
                s_dbfs_delta = 20 * np.log10(
                    s_mag_delta / (2 ** 12))  # make sure the log10 argument isn't zero (hence np.max)
                total_sum = total_sum + (s_dbfs_sum)  # sum up all the loops, then we'll average
                total_delta = total_delta + (s_dbfs_delta)  # sum up all the loops, then we'll average
            PeakValue_sum = total_sum / self.Averages
            PeakValue_delta = total_delta / self.Averages
            PeakValue_angle = total_angle / self.Averages

            if np.sign(PeakValue_angle) == -1:
                target_error = min(-0.01, (
                        np.sign(PeakValue_angle) * (PeakValue_sum - PeakValue_delta) + np.sign(
                    PeakValue_angle) * (PeakValue_sum + PeakValue_delta) / 2) / (
                                           PeakValue_sum + PeakValue_delta))
            else:
                target_error = max(0.01, (
                        np.sign(PeakValue_angle) * (PeakValue_sum - PeakValue_delta) + np.sign(
                    PeakValue_angle) * (PeakValue_sum + PeakValue_delta) / 2) / (
                                           PeakValue_sum + PeakValue_delta))

            if PeakValue_sum > max_signal:  # take the largest value, so that we know where to point the compass
                max_signal = PeakValue_sum
                max_angle = PeakValue_angle
                max_PhDelta = PhDelta
                data_fft = sum_chan
            gain.append(PeakValue_sum)
            delta.append(PeakValue_delta)
            beam_phase.append(PeakValue_angle)
            angle.append(SteerAngle)
            diff_error.append(target_error)

        NumSamples = len(data_fft)  # number of samples
        win = np.blackman(NumSamples)
        y = data_fft * win
        sp = np.absolute(np.fft.fft(y))
        sp = sp[1:-1]
        sp = np.fft.fftshift(sp)
        s_mag = np.abs(sp) * 2 / np.sum(win)  # Scale FFT by window and /2 since we are using half the FFT spectrum
        s_mag = np.maximum(s_mag, 10 ** (-15))
        max_gain = 20 * np.log10(s_mag / (2 ** 12))  # Pluto is a 12 bit ADC, so use that to convert to dBFS
        ts = 1 / float(self.rx_dev.sample_rate)
        xf = np.fft.fftfreq(NumSamples, ts)
        xf = np.fft.fftshift(xf[1:-1])  # this is the x axis (freq in Hz) for our fft plot
        # ArrayGain = gain
        # ArrayAngle = angle
        if self.gain_cal:
            return max(gain)
        if self.phase_cal:
            return angle[gain.index(min(gain))]
        if not self.gain_cal and not self.phase_cal:
            return gain, angle

    # This method starts the Gain Cal routine
    def gain_calibration(self):
        self.gain_cal = True
        for gcal_element in range(0, (4 * len(list(self.devices.values())))):
            beam_cal = [0 for i in range(0, (4 * len(list(self.devices.values()))))]
            beam_cal[gcal_element] = 0x7F
            # print(beam_cal, gcal_element)
            j = 0
            for device in self.devices.values():
                channel_gain_value = []
                for ind in range(0, 4):
                    channel_gain_value.append(beam_cal[((j * 4) + ind)])
                j += 1
                i = 0  # Gain of Individual channel
                for channel in device.channels:
                    if self.device_mode == 'rx':
                        channel.rx_gain = channel_gain_value[i]
                    elif self.device_mode == 'tx':
                        channel.tx_gain = channel_gain_value[i]
                    else:
                        raise ValueError("Configure the device first")
                    i += 1
                if self.device_mode == 'rx':
                    device.latch_rx_settings()  # writes 0x01 to reg 0x28
                elif self.device_mode == 'tx':
                    device.latch_tx_settings()  # writes 0x01 to reg 0x28
            gcal_val = self.__calculate_plot(gcal_element=gcal_element)
            self.gcalibrated_values.append(gcal_val)

        for k in range(0, 8):
            x = ((self.gcalibrated_values[k] * 127) / (min(self.gcalibrated_values)))
            self.gcal.append(int(x))
        print(self.gcal)
        self.__save_gain_cal()
        self.gain_cal = False

    # this is similar to calculate set_beam_angle
    def __set_gain_phase(self, Ph_Diff, gcal_element):
        beam_ph = [0 for i in range(0, (4 * len(list(self.devices.values()))))]
        beam_ph[gcal_element] = (np.rint(Ph_Diff * 1 / self.phase_step_size) * self.phase_step_size) % 360
        j = 0
        for device in self.devices.values():
            channel_phase_value = []
            for ind in range(0, 4):
                channel_phase_value.append(beam_ph[((j * 4) + ind)])
            j += 1
            i = 0  # Gain of Individual channel
            for channel in device.channels:
                if self.device_mode == 'rx':
                    channel.rx_gain = channel_phase_value[i]
                elif self.device_mode == 'tx':
                    channel.tx_gain = channel_phase_value[i]
                else:
                    raise ValueError("Configure the device first")
                i += 1
            if self.device_mode == 'rx':
                device.latch_rx_settings()  # writes 0x01 to reg 0x28
            elif self.device_mode == 'tx':
                device.latch_tx_settings()  # writes 0x01 to reg 0x28

    # Saves the Gain calibration values
    def __save_gain_cal(self):
        with open('gain_cal_val.pkl', 'wb') as file1:
            pickle.dump(self.gcal, file1)
            file1.close()

    # This method starts the Phase Cal routine
    def phase_calibration(self):
        self.phase_cal = True
        self.load_gain('gain_cal_val.pkl')
        for cal_element in range(0, (4*len((list(self.devices.values())))-1)):
            beam_cal = [0 for i in range(0, (4 * len(list(self.devices.values()))))]
            beam_cal[cal_element] = self.gcal[cal_element]
            beam_cal[(cal_element+1)] = self.gcal[(cal_element+1)]
            # print(beam_cal, cal_element)
            j = 0
            for device in self.devices.values():
                channel_gain_value = []
                for ind in range(0, 4):
                    channel_gain_value.append(beam_cal[((j * 4) + ind)])
                j += 1
                i = 0  # Gain of Individual channel
                for channel in device.channels:
                    if self.device_mode == 'rx':
                        channel.rx_gain = channel_gain_value[i]
                    elif self.device_mode == 'tx':
                        channel.tx_gain = channel_gain_value[i]
                    else:
                        raise ValueError("Configure the device first")
                    i += 1
                if self.device_mode == 'rx':
                    device.latch_rx_settings()  # writes 0x01 to reg 0x28
                elif self.device_mode == 'tx':
                    device.latch_tx_settings()  # writes 0x01 to reg 0x28

            cal_val = self.__calculate_plot(cal_element=cal_element)
            self.calibrated_values.append(-1 * cal_val)
        for k in range(1, len(self.calibrated_values)):
            self.calibrated_values[k] = self.calibrated_values[k] + self.calibrated_values[k - 1]

        print(self.calibrated_values)
        self.__save_phase_cal()
        self.phase_cal = False

        # this is similar to calculate plot

    # this is similar to calculate set_beam_angle
    def __set_phase_phase(self, Ph_Diff, cal_element):
        beam_ph = [0 for i in range(0, (4 * len(list(self.devices.values()))))]
        beam_ph[cal_element] = (np.rint(Ph_Diff * 1 / self.phase_step_size) * self.phase_step_size) % 360
        beam_ph[(cal_element + 1)] = ((np.rint(Ph_Diff * 1 / self.phase_step_size) * self.phase_step_size) % 360) - 180
        j = 0
        for device in self.devices.values():
            channel_phase_value = []
            for ind in range(0, 4):
                channel_phase_value.append(beam_ph[((j * 4) + ind)])
            j += 1
            i = 0  # Gain of Individual channel
            for channel in device.channels:
                if self.device_mode == 'rx':
                    channel.rx_gain = channel_phase_value[i]
                elif self.device_mode == 'tx':
                    channel.tx_gain = channel_phase_value[i]
                else:
                    raise ValueError("Configure the device first")
                i += 1
            if self.device_mode == 'rx':
                device.latch_rx_settings()  # writes 0x01 to reg 0x28
            elif self.device_mode == 'tx':
                device.latch_tx_settings()  # writes 0x01 to reg 0x28

    # Saves the Phase calibration values
    def __save_phase_cal(self):
        with open('phase_cal_val.pkl', 'wb') as file:
            pickle.dump(self.calibrated_values, file)
            file.close()

    # Gain plots sum_chan. Delta plots the difference and Error plots the diff of sum & delta chans
    def plot(self):
        gain, angle = self.__calculate_plot()
        plt.clf()
        # plt.plot(xf/1e6, max_gain)
        plt.scatter(gain, angle)
        plt.show()

class pll:
    def __init__(self):
        super(pll, self).__init__()

    class adf4159(adf4159):
        def __init__(self, uri):
            adf4159.__init__(self, uri)

        def configure(self):
            self.frequency = 6247500000
            self.freq_dev_step = 5690
            self.freq_dev_range = 0
            self.freq_dev_time = 0
            self.powerdown = 0
            self.ramp_mode = "disabled"

    class pluto_pll(ad9361):
        def __init__(self, uri):
            ad9361.__init__(self, uri)

        def configure(self):
            self.tx_lo = 6000000000
            self.tx_cyclic_buffer = True
            self._tx_buffer_size = int(2 ** 18)
            self.rx_lo = 4495000000  # Recieve Freq
            # Make sure the Tx channels are attenuated (or off) and their freq is far away from Rx
            self.tx_hardwaregain_chan0 = -10
            self.tx_hardwaregain_chan1 = -10


class sdr(ad9361):
    def __init__(self, uri):
        ad9361.__init__(self, uri)

    def configure(self):
        self._ctrl.debug_attrs["adi,frequency-division-duplex-mode-enable"].value = "1"
        self._ctrl.debug_attrs[
            "adi,ensm-enable-txnrx-control-enable"].value = "0"  # Disable pin control so spi can move the states
        self._ctrl.debug_attrs["initialize"].value = "1"
        self.rx_enabled_channels = [0, 1]  # enable Rx1 (voltage0) and Rx2 (voltage1)
        self.gain_control_mode_chan1 = 'manual'  # We must be in manual gain control mode (otherwise we won't see
        self.rx_hardwaregain_chan1 = 40
        self._rxadc.set_kernel_buffers_count(1)
        rx = self._ctrl.find_channel('voltage0')
        rx.attrs['quadrature_tracking_en'].value = '1'  # set to '1' to enable quadrature tracking
        self.sample_rate = int(2000000)  # Sampling rate
        self.rx_buffer_size = int(4 * 256)
        self.rx_rf_bandwidth = int(10e6)
        # We must be in manual gain control mode (otherwise we won't see the peaks and nulls!)
        self.gain_control_mode_chan0 = 'manual'
        self.rx_hardwaregain_chan0 = 40
        self.rx_lo = 2000000000  # 4495000000  # Recieve Freq
