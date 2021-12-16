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

from adi.adar1000 import adar1000_array
import numpy as np
import pickle
import matplotlib.pyplot as plt

''' 
    Phased Array beamformwer has 3 main devices 1. beamformer 2. Rx_device - SDR 3. Tx_device - pll
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
        self.SignalFreq = 10492000000  # Frequency of source
        self.Averages = 1  # Number of Avg to be taken. We be user input value on GUI
        self.phase_step_size = 2.8125  # it is 360/2**number of bits. For now number of bits == 6. change later
        self.steer_res = 2.8125  # It is steering resolution. This would be user selected value
        self.c = 299792458  # speed of light in m/s
        self.element_spacing = 0.015  # element to element spacing of the antenna
        self.res_bits = 1  # res_bits and bits are two different var. It can be variable, but it is hardset to 1 for now
        self.pcal = [0 for i in range(0, (4 * len(list(self.devices.values()))))]  # default phase cal value i.e 0
        self.gcal = [0x7F for i in range(0, 4 * len(list(self.devices.values())))]  # default gain cal value i.e 127
        self.rx_dev = None  # rx_device/sdr that rx and plots
        self.gain_cal = False  # gain/phase calibration status flag it goes True when performing calibration
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
            # False sets a bit high and SPI control
            device.beam_mem_enable = False  # RAM control vs SPI control of the adar state, reg 0x38, bit 6.
            device.bias_mem_enable = False  # RAM control vs SPI control of the bias state, reg 0x38, bit 5.
            device.pol_state = False  # Polarity switch state, reg 0x31, bit 0. True outputs -5V, False outputs 0V
            device.pol_switch_enable = False  # Enables switch driver for ADTR1107 switch, reg 0x31, bit 3
            device.tr_source = 'spi'  # TR source for chip, reg 0x31 bit 2. 'ext' sets bit high, 'spi' sets a bit low
            device.tr_spi = 'rx'  # TR SPI control, reg 0x31 bit 1.  'tx' sets bit high, 'rx' sets a bit low
            device.tr_switch_enable = True  # Switch driver for external switch, reg0x31, bit 4
            device.external_tr_polarity = True  # Sets polarity of TR switch compared to TR state of ADAR1000.

            device.rx_vga_enable = True  # Enables Rx VGA, reg 0x2E, bit 0.
            device.rx_vm_enable = True  # Enables Rx VGA, reg 0x2E, bit 1.
            device.rx_lna_enable = True  # Enables Rx LNA, reg 0x2E, bit 2. bit3,4,5,6 enables RX for all the channels
            device._ctrl.reg_write(0x2E, 0x7F)  # bit3,4,5,6 enables RX for all the channels.
            device.rx_lna_bias_current = 8  # Sets the LNA bias to the middle of its range
            device.rx_vga_vm_bias_current = 22  # Sets the VGA and vector modulator bias.

            device.tx_vga_enable = True  # Enables Tx VGA, reg 0x2F, bit0
            device.tx_vm_enable = True  # Enables Tx Vector Modulator, reg 0x2F, bit1
            device.tx_pa_enable = True  # Enables Tx channel drivers, reg 0x2F, bit2
            device.tx_pa_bias_current = 6  # Sets Tx driver bias current
            device.tx_vga_vm_bias_current = 22  # Sets Tx VGA and VM bias.

            if self.device_mode == "rx":
                # Configure the device for Rx mode
                device.mode = "rx"  # Mode of operation, bit 5 of reg 0x31. "rx", "tx", or "disabled".

                SELF_BIASED_LNAs = True
                if SELF_BIASED_LNAs:
                    # Allow the external LNAs to self-bias
                    # this writes 0xA0 0x30 0x00. Disabling it allows LNAs to stay in self bias mode all the time
                    device.lna_bias_out_enable = False
                    # self._ctrl.reg_write(0x30, 0x00)   #Disables PA and DAC bias
                else:
                    # Set the external LNA bias
                    device.lna_bias_on = -0.7  # this writes 0x25 to register 0x2D.
                    # self._ctrl.reg_write(0x30, 0x20)   #Enables PA and DAC bias.

                # Enable the Rx path for each channel
                for channel in device.channels:
                    channel.rx_enable = True  # this writes reg0x2E with data 0x00, then reg0x2E with data 0x20.
                    #  So it overwrites 0x2E, and enables only one channel

            # Configure the device for Tx mode
            elif self.device_mode == "tx":
                device.mode = "tx"

                # Enable the Tx path for each channel and set the external PA bias
                for channel in device.channels:
                    channel.tx_enable = True
                    channel.pa_bias_on = -2

            else:
                raise ValueError("Configure Device in proper mode")  # If device mode is neither Rx nor Tx

            if self.device_mode == "rx":
                device.latch_rx_settings()  # writes 0x01 to reg 0x28.
            elif self.device_mode == "tx":
                device.latch_tx_settings()  # writes 0x02 to reg 0x28.

    # This method load gain calibrated value from filepath specified, if not calibrated set all channel gain to max
    def load_gain_cal(self, filename='', calibration=False):
        """I used pickle file to serialize the data. Maybe could switch to more standard format like JSON or csv
            calibration is a flag so when phase calibration is loading gain values it is set and does not return
            default values if it fails and throws error"""
        if not calibration:
            try:
                with open(filename, 'rb') as file1:
                    self.gcal = pickle.load(file1)  # Load gain cal values
            except:
                for i in range(0, (4 * len(list(self.devices.values())))):  # if it fails load default value
                    self.gcal.append(0x7F)

        elif calibration:  # When Phase calibration
            try:
                with open(filename, 'rb') as file1:
                    self.gcal = pickle.load(file1)  # Load gain cal values
                    calibration = False  # Reset calibration flag, not req practically but to be sure.
            except:
                calibration = False
                raise SystemError("Perform Gain calibration first")  # if it fails raise error

    # This method load phase calibrated value from filepath specified, if not cali set all channel phase correction to 0
    def load_phase_cal(self, filename=''):
        try:
            with open(filename, 'rb') as file:
                self.pcal = pickle.load(file)  # Load gain cal values
        except:
            for i in range(0, (4 * len(list(self.devices.values())))):
                self.pcal.append(0)  # if it fails load default value i.e. 0

    # This will try to set all the gain to it's calibrated values. If system is not calibrated set all chans to max gain
    def set_all_gain(self):
        j = 0  # j is index of device and device indicate the adar1000 on which operation is currently done
        for device in self.devices.values():  # device in dict of all adar1000 connected
            channel_gain_value = []  # channel gain value to be written on ind channel
            for ind in range(0, 4):  # ind is index of current channel of current device
                channel_gain_value.append(self.gcal[((j * 4) + ind)])
            j += 1
            i = 0  # i is index of channel of each device
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

    # This allows to set individual gain of the channel
    def set_chan_gain(self, chan_no, gain_val):  # chan_no is the ch whose gain you want to set, gain_val is value
        """Each device has 4 channels but for top level channel numbers are 1 to 8 so took dev as Quotient of
           channel num div by 4 and channel of that dev is overall chan num minus 4 x that dev number"""
        if self.device_mode == 'rx':
            list(self.devices.values())[chan_no // 4].channels[(chan_no - (4 * (chan_no // 4)))].rx_gain = gain_val
        elif self.device_mode == 'tx':
            list(self.devices.values())[chan_no // 4].channels[(chan_no - (4 * (chan_no // 4)))].tx_gain = gain_val
        else:
            raise ValueError("Configure the device first")

    """ A public method to sweep the phase value from -180 to 180 deg, calculate phase values of all the channel
      and set them. If we want beam angle at fixed angle you can pass angle value at which you want center lobe"""
    def set_beam_angle(self, Ph_Diff):
        j = 0  # j is index of device and device indicate the adar1000 on which operation is currently done
        for device in list(self.devices.values()):  # device in dict of all adar1000 connected
            channel_phase_value = []  # channel phase value to be written on ind channel
            for ind in range(0, 4):  # ind is index of current channel of current device
                channel_phase_value.append(((np.rint(Ph_Diff * ((j * 4) + ind) / self.phase_step_size) *
                                             self.phase_step_size) + self.pcal[((j * 4) + ind)]) % 360)
            j += 1
            i = 0  # i is index of channel of each device
            for channel in device.channels:
                # Set phase depending on the device mode
                if self.device_mode == "rx":
                    channel.rx_phase = channel_phase_value[
                        i]  # writes to I and Q registers values according to Table 13-16 from datasheet.
                i = i + 1
            if self.device_mode == "rx":
                device.latch_rx_settings()
            else:
                device.latch_tx_settings()

    """ This method calculate all the values required to do different plots. It method calls set_beam_angle and 
        sets the Phases of all channel. All the math is done here"""
    def __calculate_plot(self, gcal_element=0, cal_element=0):
        # These are all the phase deltas (i.e. phase difference between Rx1 and Rx2, then Rx2 and Rx3, etc.) we'll sweep
        PhaseValues = np.arange(-196.875, 196.875, self.phase_step_size)
        max_signal = -100000  # Reset max_signal.  We'll keep track of the maximum signal we get as we do this 140 loop.
        max_angle = -90  # Reset max_angle. This is the angle where we saw the max signal.
        gain, delta, beam_phase, angle, diff_error = [], [], [], [], []  # Create empty lists
        for PhDelta in PhaseValues:  # These sweeps phase value from -180 to 180
            # set Phase of channels based on Calibration Flag status and calibration element
            if self.gain_cal:
                self.__set_gain_phase(PhDelta, gcal_element)
            if self.phase_cal:
                self.__set_phase_phase(PhDelta, cal_element)
            if not self.gain_cal and not self.phase_cal:
                self.set_beam_angle(PhDelta)
            # arcsin argument must be between 1 and -1, or numpy will throw a warning
            if PhDelta >= 0:
                SteerAngle = np.degrees(np.arcsin(max(min(1, (self.c * np.radians(np.abs(PhDelta))) / (
                        2 * 3.14159 * self.SignalFreq * self.element_spacing)),
                                                      -1)))  # positive PhaseDelta covers 0deg to 90 deg
            else:
                SteerAngle = -(np.degrees(np.arcsin(max(min(1, (self.c * np.radians(np.abs(PhDelta))) / (
                        2 * 3.14159 * self.SignalFreq * self.element_spacing)),
                                                        -1))))  # negative phase delta covers 0 deg to -90 deg

            total_sum, total_delta, total_angle = 0, 0, 0
            for count in range(0, self.Averages):  # repeat loop and average the results
                data = self.rx_dev.rx()  # read a buffer of data from Pluto using pyadi-iio library (adi.py)
                N = len((data[0] + data[1]))  # number of samples len(sum_chan) = 1 as just 1st element of list is taken
                win = np.blackman(N)
                y_sum = (data[0] + data[1]) * win
                y_delta = (data[0] - data[1]) * win
                s_sum = np.fft.fftshift(np.absolute(np.fft.fft(y_sum))[1:-1])
                s_delta = np.fft.fftshift(np.absolute(np.fft.fft(y_delta))[1:-1])
                total_angle = total_angle + (np.angle(s_sum[np.argmax(s_sum)]) - np.angle(s_delta[np.argmax(s_sum)]))
                s_mag_sum = np.maximum(np.abs(s_sum[np.argmax(s_sum)]) * 2 / np.sum(win), 10 ** (-15))
                s_mag_delta = np.maximum(np.abs(s_delta[np.argmax(s_sum)]) * 2 / np.sum(win), 10 ** (-15))
                total_sum = total_sum + (20 * np.log10(s_mag_sum / (2 ** 12)))  # sum up all the loops, then we'll avg
                total_delta = total_delta + (20 * np.log10(s_mag_delta / (2 ** 12)))
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
                data_fft = (data[0] + data[1])
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
        # Return values/ parameter based on Calibration Flag status
        if self.gain_cal:
            return max(gain)
        if self.phase_cal:
            return angle[gain.index(min(gain))]
        if not self.gain_cal and not self.phase_cal:
            return gain, angle, delta, diff_error, beam_phase, xf, max_gain

    # This method starts the Gain Cal routine
    def gain_calibration(self):
        self.gain_cal = True  # Gain Calibration Flag
        self.gcal = []  # Reset the initiated gcal array so that appending does not result in extra values
        gcalibrated_values = []  # Intermediate cal values list
        # gcal_element indicates current element/channel which is being calibrated
        for gcal_element in range(0, (4 * len(list(self.devices.values())))):
            beam_cal = [0 for i in range(0, (4 * len(list(self.devices.values()))))]  # set all gain to 0
            beam_cal[gcal_element] = 0x7F  # Only set th gain of current element/channel to max
            # print(beam_cal, gcal_element)
            j = 0  # # j is index of device and device indicate the adar1000 on which operation is currently done
            # Note operations and explanation is similar to set_all_gain and set_beam_angle
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
            gcal_val = self.__calculate_plot(gcal_element=gcal_element)  # cal plot according to element and routine
            gcalibrated_values.append(gcal_val)  # make a list of intermediate cal values

        """Minimum gain of intermediated cal val is set to Maximum value as we cannot go beyond max value and gain
           of all other channels are set accordingly"""
        for k in range(0, 8):
            x = ((gcalibrated_values[k] * 127) / (min(gcalibrated_values)))
            self.gcal.append(int(x))  # append final calibrated value to gcal list

        # print(self.gcal)
        self.__save_gain_cal()  # save the gcal list
        self.gain_cal = False  # Reset the Gain calibration Flag once system gain is calibrated

    # This method sets phase of channel when gain calibration is taking place
    def __set_gain_phase(self, Ph_Diff, gcal_element):
        beam_ph = [0 for i in range(0, (4 * len(list(self.devices.values()))))]
        beam_ph[gcal_element] = (np.rint(Ph_Diff * 1 / self.phase_step_size) * self.phase_step_size) % 360
        # Note operations and explanation is similar to set_all_gain and set_beam_angle
        j = 0
        for device in self.devices.values():
            channel_phase_value = []
            for ind in range(0, 4):
                channel_phase_value.append(beam_ph[((j * 4) + ind)])
            j += 1
            i = 0  # Gain of Individual channel
            for channel in device.channels:
                if self.device_mode == 'rx':
                    channel.rx_phase = channel_phase_value[i]
                elif self.device_mode == 'tx':
                    channel.tx_phase = channel_phase_value[i]
                else:
                    raise ValueError("Configure the device first")
                i += 1
            if self.device_mode == 'rx':
                device.latch_rx_settings()  # writes 0x01 to reg 0x28
            elif self.device_mode == 'tx':
                device.latch_tx_settings()  # writes 0x01 to reg 0x28

    # This method automatically saves gain cal and need not to be accessed from top layer
    def __save_gain_cal(self):
        with open('gain_cal_val.pkl', 'wb') as file1:
            pickle.dump(self.gcal, file1)  # save calibrated gain value to a file
            file1.close()

    # This method starts the Phase Cal routine
    def phase_calibration(self):
        self.phase_cal = True  # Gain Calibration Flag
        self.load_gain_cal('gain_cal_val.pkl', True)  # Load gain cal val as phase cal is dependent on gain cal
        self.pcal = []  # Reset the initiated pcal array so that appending does not result in extra values
        # cal_element indicates current element/channel which is being calibrated
        for cal_element in range(0, (4 * len((list(self.devices.values()))) - 1)):
            beam_cal = [0 for i in range(0, (4 * len(list(self.devices.values()))))]  # set all gain to 0
            # Only set th gain of current & it's adjacent element/channel to gain calibrated values
            beam_cal[cal_element] = self.gcal[cal_element]
            beam_cal[(cal_element + 1)] = self.gcal[(cal_element + 1)]
            # Note operations and explanation is similar to set_all_gain and set_beam_angle
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

            cal_val = self.__calculate_plot(cal_element=cal_element)  # cal plot according to element and routine
            self.pcal.append(-1 * cal_val)  # append values to calibrated value list
        """All the channels are calibrated with respect to it's next/adjacent channel. Map back all the cal values to
           a single channel. Here everything is maped to 1st channel"""
        for k in range(1, len(self.pcal)):
            self.pcal[k] = self.pcal[k] + self.pcal[k - 1]
        self.pcal.insert(0, 0)  # Calibration of 1st channel is 0 w.r.t itself so add 0 in start on phase cal list
        self.__save_phase_cal()  # save the pcal list
        self.phase_cal = False  # Reset the Phase calibration Flag once system phase is calibrated

    # This method sets phase of channel when phase calibration is taking place
    def __set_phase_phase(self, Ph_Diff, cal_element):
        # Note operations and explanation is similar to set_all_gain and set_beam_angle
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
                    channel.rx_phase = channel_phase_value[i]
                elif self.device_mode == 'tx':
                    channel.tx_phase = channel_phase_value[i]
                else:
                    raise ValueError("Configure the device first")
                i += 1
            if self.device_mode == 'rx':
                device.latch_rx_settings()  # writes 0x01 to reg 0x28
            elif self.device_mode == 'tx':
                device.latch_tx_settings()  # writes 0x01 to reg 0x28

    # This method automatically saves phase cal and need not to be accessed from top layer
    def __save_phase_cal(self):
        with open('phase_cal_val.pkl', 'wb') as file:
            pickle.dump(self.pcal, file)  # save calibrated phase value to a file
            file.close()

    # Gain plots sum_chan. Delta plots the difference and Error plots the diff of sum & delta chans
    def plot(self, plot_type: str):
        gain, angle, delta, diff_error, beam_phase, xf, max_gain = self.__calculate_plot()
        if plot_type == "fft":
           return xf / 1e6, max_gain
        elif plot_type == "monopulse":
            return gain, angle
