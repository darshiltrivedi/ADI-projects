import adi
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

sys.path.append('C:\ProgramData\Anaconda3\envs\adi\Lib\site-packages')
# import iio

def ad9361_init(pluto_tx):
    # Setup Pluto
    pluto_tx._ctrl.debug_attrs[
        "adi,frequency-division-duplex-mode-enable"].value = "1"  # move to fdd mode.  see https://github.com/analogdevicesinc/pyadi-iio/blob/ensm-example/examples/ad9361_advanced_ensm.py
    pluto_tx._ctrl.debug_attrs[
        "adi,ensm-enable-txnrx-control-enable"].value = "0"  # Disable pin control so spi can move the states
    pluto_tx._ctrl.debug_attrs["initialize"].value = "1"
    pluto_tx.rx_enabled_channels = [0, 1]  # enable Rx1 (voltage0) and Rx2 (voltage1)
    pluto_tx.gain_control_mode_chan0 = 'manual'  # We must be in manual gain control mode (otherwise we won't see the peaks and nulls!)
    pluto_tx.gain_control_mode_chan1 = 'manual'  # We must be in manual gain control mode (otherwise we won't see the peaks and nulls!)


    pluto_tx._rxadc.set_kernel_buffers_count(
        1)  # Default is 4 Rx buffers are stored, but we want to change and immediately measure the result, so buffers=1
    rx = pluto_tx._ctrl.find_channel('voltage0')
    rx.attrs['quadrature_tracking_en'].value = '1'  # set to '1' to enable quadrature tracking
    pluto_tx.sample_rate = 3000000  # Sampling rate
    pluto_tx.rx_buffer_size = int(4 * 256)
    pluto_tx.tx_cyclic_buffer = True  # start the cyclic buffer
    pluto_tx.tx_lo = 6000000000  # Transmit freq
    pluto_tx.tx_buffer_size = int(2 ** 18)

    pluto_tx.tx_hardwaregain_chan0 = -80  # Make sure the Tx channels are attenuated (or off) and their freq is far away from Rx
    pluto_tx.tx_hardwaregain_chan1 = -80
    pluto_tx.rx_hardwaregain_chan0 = 30
    pluto_tx.rx_hardwaregain_chan1 = 30

    # pluto_tx.tx_hardwaregain_chan0 = -80  # this is a negative number between 0 and -88
    # pluto_tx.gain_control_mode_chan0 = "manual"  # We must be in manual gain control mode (otherwise we won't see the peaks and nulls!)
    # pluto_tx.rx_hardwaregain_chan0 = 30

    # pluto_tx.dds_enabled = [1, 1, 1, 1]                  #DDS generator enable state
    # pluto_tx.dds_frequencies = [0.1e6, 0.1e6, 0.1e6, 0.1e6]      #Frequencies of DDSs in Hz
    # pluto_tx.dds_scales = [1, 1, 0, 0]                   #Scale of DDS signal generators Ranges [0,1]
    pluto_tx.dds_single_tone(int(0.0e6), 0.9, 0)    # pluto_tx.dds_single_tone(tone_freq_hz, tone_scale_0to1, tx_channel)

    # pluto_tx._tx_buffer_size = 2 ** 13  # set size of transmit buffer
    # pluto_tx.rx_buffer_size = 2 ** 13  # set size of recieve buffer
    # pluto_tx.tx_rf_bandwidth = 55000000  # Bandwidth of Rx and Tx

    pluto_tx.rx_lo = 4492000000  # Recieve Freq
    # pluto_tx.tx_hardwaregain_chan0 = -80
    # # pluto_tx.rx_hardwaregain_chan0 = 80
    # pluto_tx.gain_control_mode_chan0 = 'manual'
    # pluto_tx.dds_scales = ('0.3', '0.25', '0', '0')  # scale the dds tone, value should be btw 0 and 1
    # pluto_tx.dds_enabled = (1, 1, 1, 1)  # Enable dds on different channels.  1 = enable and 0 = disable
    # pluto_tx.filter = "LTE20_MHz.ftr"  # Apply filter to the transreciever. File should be locally available or provide the path from IIO scope directory.
    time.sleep(0.1)
    # data = pluto_tx.rx()
    # print(data)
    # Pluto_Rx(pluto_tx)


def PLUTO_init(pluto_tx):
    pluto_tx._rxadc.set_kernel_buffers_count(1)  # Default is 4 Rx buffers are stored, but we want to change and immediately measure the result, so buffers=1
    rx = pluto_tx._ctrl.find_channel('voltage0')
    rx.attrs['quadrature_tracking_en'].value = '1'  # set to '1' to enable quadrature tracking
    pluto_tx.sample_rate = int(30000000) # Jon has 40MHz but it throws error on pluto
    # pluto_txfilter = "/home/pi/Documents/PlutoFilters/samprate_40p0.ftr"  #pyadi-iio auto applies filters based on sample rate
    # pluto_tx.rx_rf_bandwidth = int(40e6)
    # pluto_tx.tx_rf_bandwidth = int(40e6)
    pluto_tx.rx_buffer_size = int(2**10)
    pluto_tx.tx_lo = 6000000000
    pluto_tx.tx_cyclic_buffer = True
    pluto_tx._tx_buffer_size = int(2 ** 18)

    pluto_tx.tx_hardwaregain_chan0 = -3  # this is a negative number between 0 and -88
    pluto_tx.gain_control_mode_chan0 = "manual"  # We must be in manual gain control mode (otherwise we won't see the peaks and nulls!)
    pluto_tx.rx_hardwaregain_chan0 = 60

    # pluto_tx.dds_enabled = [1, 1, 1, 1]                  #DDS generator enable state
    # pluto_tx.dds_frequencies = [0.1e6, 0.1e6, 0.1e6, 0.1e6]      #Frequencies of DDSs in Hz
    # pluto_tx.dds_scales = [1, 1,pinf 0, 0]                   #Scale of DDS signal generators Ranges [0,1]
    pluto_tx.dds_single_tone(int(0.0e6), 0.9, 0)  # pluto_tx.dds_single_tone(tone_freq_hz, tone_scale_0to1, tx_channel)
    pluto_tx.rx_lo = 4492000000


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


def ADAR_init(beam, adar_mode):
    # Configure ADAR1000
    # beam.initialize_devices()  # Always Intialize the device 1st as reset is performed at Initialization
    # If ADAR1000 array is used initialization work other wise reset each adar individually
    beam.reset()  # Performs a soft reset of the device (writes 0x81 to reg 0x00)
    time.sleep(1)
    beam._ctrl.reg_write(0x000,0x18)
    beam._ctrl.reg_write(0x400, 0x55)  # This trims the LDO value to approx. 1.8V (to the center of its range)

    beam.sequencer_enable = False
    beam.beam_mem_enable = False  # RAM control vs SPI control of the beam state, reg 0x38, bit 6.  False sets bit high and SPI control
    # beam.bias_mem_enable = False  # RAM control vs SPI control of the bias state, reg 0x38, bit 5.  False sets bit high and SPI control
    beam.pol_state = False  # Polarity switch state, reg 0x31, bit 0. True outputs -5V, False outputs 0V
    beam.pol_switch_enable = False  # Enables switch driver for ADTR1107 switch, reg 0x31, bit 3
    beam.tr_source = 'spi'  # TR source for chip, reg 0x31 bit 2.  'external' sets bit high, 'spi' sets bit low
    beam.tr_spi = 'rx'  # TR SPI control, reg 0x31 bit 1.  'tx' sets bit high, 'rx' sets bit low
    beam.tr_switch_enable = True  # Switch driver for external switch, reg0x31, bit 4
    beam.external_tr_polarity = True  # Sets polarity of TR switch compared to TR state of ADAR1000.  True outputs 0V in Rx mode

    beam.rx_vga_enable = True  # Enables Rx VGA, reg 0x2E, bit 0.
    beam.rx_vm_enable = True  # Enables Rx VGA, reg 0x2E, bit 1.
    beam.rx_lna_enable = True  # Enables Rx LNA, reg 0x2E, bit 2.
    beam.rx_lna_bias_current = 8  # Sets the LNA bias to the middle of its range
    beam.rx_vga_vm_bias_current = 22  # Sets the VGA and vector modulator bias.  I thought this should be 22d, but Stingray has it as 85d????

    beam.tx_vga_enable = True  # Enables Tx VGA, reg 0x2F, bit0
    beam.tx_vm_enable = True  # Enables Tx Vector Modulator, reg 0x2F, bit1
    beam.tx_pa_enable = True  # Enables Tx channel drivers, reg 0x2F, bit2
    beam.tx_pa_bias_current = 6  # Sets Tx driver bias current
    beam.tx_vga_vm_bias_current = 22  # Sets Tx VGA and VM bias.  I thought this should be 22d, but stingray has as 45d??????
    # print("Done")

    if adar_mode == "rx":
        # Configure the device for Rx mode
        beam.mode = "rx"  # Mode of operation, bit 5 of reg 0x31. "rx", "tx", or "disabled"
        print(beam.mode)
        SELF_BIASED_LNAs = True
        if SELF_BIASED_LNAs:
            # Allow the external LNAs to self-bias
            beam.lna_bias_out_enable = False  # this writes 0xA0 0x30 0x00.  What does this do? # Disabling it allows LNAs to stay in self bias mode all the time
            # beam._ctrl.reg_write(0x30, 0x00)   #Disables PA and DAC bias
        else:
            # Set the external LNA bias
            beam.lna_bias_on = -0.7  # this writes 0x25 to register 0x2D.  This is correct.  But oddly enough, it doesn't first write 0x18 to reg 0x00....
            # beam._ctrl.reg_write(0x30, 0x20)   #Enables PA and DAC bias.  I think this would be needed too?

        # Enable the Rx path for each channel
        for channel in beam.channels:
            channel.rx_enable = True  # this writes reg0x2E with data 0x00, then reg0x2E with data 0x20.  So it overwrites 0x2E, and enables only one channel
            # beam._ctrl.reg_write(0x2E, 0x7F)    #Enables all four Rx channels, the Rx LNA, Rx Vector Modulator and Rx VGA

    # Configure the device for Tx mode
    else:
        beam.mode = "tx"

        # Enable the Tx path for each channel and set the external PA bias
        for channel in beam.channels:
            channel.tx_enable = True
            channel.pa_bias_on = -2

    if adar_mode == "rx":
        beam.latch_rx_settings()  # writes 0x01 to reg 0x28.
    else:
        beam.latch_tx_settings()  # writes 0x02 to reg 0x28.
    print(beam)

def ADAR_set_RxTaper(beam, adar_mode):
    # Set the gains to max gain (0x7f, or 127)
    for channel in beam.channels:
        # print(channel)
        # Set the gain and phase depending on the device mode
        if adar_mode == "rx":
            channel.rx_gain = 0x7F
            time.sleep(0.1)
        else:
            channel.tx_gain = 0x67

    beam._ctrl.reg_write(0x2010, 0xFF)

    print(beam.channel1.rx_gain)


    x = beam.channel2.rx_gain
    print(x)

    # Latch in the new gains.
    if adar_mode == "rx":
        beam.latch_rx_settings()  # writes 0x01 to reg 0x28.
    else:
        beam.latch_tx_settings()  # writes 0x02 to reg 0x28.



def ADAR_set_RxPhase(beam, adar_mode, ADAR_ADD, Ph_Diff, ph_step_size):
    Rx1_Phase_Cal = 0
    Rx2_Phase_Cal = 0
    Rx3_Phase_Cal = 0
    Rx4_Phase_Cal = 0
    Rx5_Phase_Cal = 0
    Rx6_Phase_Cal = 0
    Rx7_Phase_Cal = 0
    Rx8_Phase_Cal = 0
    # print(Ph_Diff)
    if ADAR_ADD == 1:
        Phase_A = ((np.rint(Ph_Diff * 0 / ph_step_size) * ph_step_size) + Rx1_Phase_Cal) % 360  # round each value to the nearest step size increment
        Phase_B = ((np.rint(Ph_Diff * 1 / ph_step_size) * ph_step_size) + Rx2_Phase_Cal) % 360
        Phase_C = ((np.rint(Ph_Diff * 2 / ph_step_size) * ph_step_size) + Rx3_Phase_Cal) % 360
        Phase_D = ((np.rint(Ph_Diff * 3 / ph_step_size) * ph_step_size) + Rx4_Phase_Cal) % 360

    elif ADAR_ADD == 2:
        Phase_A = ((np.rint(Ph_Diff * 4 / ph_step_size) * ph_step_size) + Rx5_Phase_Cal) % 360
        Phase_B = ((np.rint(Ph_Diff * 5 / ph_step_size) * ph_step_size) + Rx6_Phase_Cal) % 360
        Phase_C = ((np.rint(Ph_Diff * 6 / ph_step_size) * ph_step_size) + Rx7_Phase_Cal) % 360
        Phase_D = ((np.rint(Ph_Diff * 7 / ph_step_size) * ph_step_size) + Rx8_Phase_Cal) % 360
    channel_phase_value = [Phase_A, Phase_B, Phase_C, Phase_D]
    # print([Phase_A, Phase_B, Phase_C, Phase_D])

    # print(channel_phase_value)

    beam.channel1.rx_phase = channel_phase_value[0]
    beam.channel2.rx_phase = channel_phase_value[1]
    beam.channel3.rx_phase = channel_phase_value[2]
    beam.channel4.rx_phase = channel_phase_value[3]

    # i = 0
    # for channel in beam.channels:
    #     # Set phase depending on the device mode
    #     if adar_mode == "rx":
    #         channel.rx_phase = channel_phase_value[i]  # writes to I and Q registers values according to Table 13-16 from datasheet.
    #     i = i + 1
    beam.latch_rx_settings()


    return None


def ADAR_Plotter(beam_list, device_mode, sdr):
    x = 1  # This is for now toggle value between Pluto and ad9361. Pluto == 1(1r1t) and ad9361 == 2(2r2t)
    SignalFreq = 10492000000  # Frequency of source
    Averages = 1  # Number of Avg too be taken. We be user input value on GUI
    phase_step_size = 2.8125  # it is 360/2**number of bits. For now number of bits == 6. change later
    steer_res = 2.8125  # It is steering resolution. This would be user selected value
    c = 299792458  # speed of light in m/s
    d = 0.015  # element to element spacing of the antenna
    num_ADARs = 1
    res_bits = 1  # res_bits and bits are two different vaiable. It can be variable but it is hardset to 1 for now
    if x == 2:
        while True:

            PhaseValues = np.arange(-196.875, 196.875,
                                    phase_step_size)  # These are all the phase deltas (i.e. phase difference between Rx1 and Rx2, then Rx2 and Rx3, etc.) we'll sweep.
            PhaseStepNumber = 0  # this is the number of phase steps we'll take (140 in total).  At each phase step, we set the individual phases of each of the Rx channels
            max_signal = -100  # Reset max_signal.  We'll keep track of the maximum signal we get as we do this 140 loop.
            max_angle = 0  # Reset max_angle.  This is the angle where we saw the max signal.  This is where our compass will point.
            output_items = np.zeros((5, 140), dtype=complex)

            for PhDelta in PhaseValues:
                for i in range(0, 0):  # change according to number of adar1000 connected
                    ADAR_set_RxPhase(beam_list[i], device_mode, num_ADARs, PhDelta, phase_step_size)

                value1 = (c * np.radians(np.abs(PhDelta))) / (2 * 3.14159 * SignalFreq * d)
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
                for count in range(0, Averages):  # repeat loop and average the results
                    data = sdr.rx()  # read a buffer of data from Pluto using pyadi-iio library (adi.py)
                    chan1 = data[0]  # Rx1 data. data is a list of values chan1 and chan2 are just 1st and 2nd element of list Do we have to discard all other values?
                    chan2 = data[1]  # Rx2 data. Changing data[0] to data. delta chan is all 0.
                    # print(chan1, chan2, data)
                    sum_chan = chan1 + chan2
                    delta_chan = chan1 - chan2
                    # print(sum_chan, delta_chan)
                    N = len(sum_chan)  # number of samples  len(sum_chan) = 1 as just 1st element of list is taken
                    win = np.hamming(N)
                    y_sum = sum_chan * win
                    y_delta = delta_chan * win
                    s_sum = np.fft.fftshift(y_sum)
                    s_delta = np.fft.fftshift(y_delta)
                    max_index = np.argmax(s_sum)
                    total_angle = total_angle + (np.angle(s_sum[max_index]) - np.angle(s_delta[max_index]))

                    s_mag_sum = np.abs(s_sum[max_index])  # * 2 / np.sum(win)
                    s_mag_delta = np.abs(s_delta[max_index])  # * 2 / np.sum(win)
                    s_dbfs_sum = 20 * np.log10(
                        np.max([s_mag_sum, 10 ** (-15)]) / (2 ** 12))  # make sure the log10 argument isn't zero (hence np.max)
                    s_dbfs_delta = 20 * np.log10(np.max([s_mag_delta, 10 ** (-15)]) / (
                            2 ** 12))  # make sure the log10 argument isn't zero (hence np.max)
                    total_sum = total_sum + (s_dbfs_sum)  # sum up all the loops, then we'll average
                    total_delta = total_delta + (s_dbfs_delta)  # sum up all the loops, then we'll average
                PeakValue_sum = total_sum / Averages
                PeakValue_delta = total_delta / Averages
                PeakValue_angle = total_angle / Averages

                if PeakValue_angle > 0:
                    diff_angle = 1
                    diff_chan = max((diff_angle * (PeakValue_sum - PeakValue_delta) + diff_angle * (
                            PeakValue_sum + PeakValue_delta) / 2) / (PeakValue_sum + PeakValue_delta), 0.01)
                else:
                    diff_angle = -1
                    diff_chan = min((diff_angle * (PeakValue_sum - PeakValue_delta) + diff_angle * (
                            PeakValue_sum + PeakValue_delta) / 2) / (PeakValue_sum + PeakValue_delta), -0.01)
                # diff_angle=PeakValue_angle
                # diff_chan = (diff_angle * (PeakValue_sum - PeakValue_delta) + diff_angle * (PeakValue_sum + PeakValue_delta)/2) / (PeakValue_sum + PeakValue_delta)
                # diff_chan = PeakValue_sum - PeakValue_delta

                if PeakValue_sum > max_signal:  # take the largest value, so that we know where to point the compass
                    max_signal = PeakValue_sum
                    max_angle = PhDelta

                output_items[0][PhaseStepNumber] = ((1) * SteerAngle + (
                        1j * PeakValue_sum))  # output this as a complex number so we can do an x-y plot with the constellation graph
                output_items[1][PhaseStepNumber] = ((1) * SteerAngle + (
                        1j * PeakValue_delta))  # output this as a complex number so we can do an x-y plot with the constellation graph
                output_items[2][PhaseStepNumber] = ((1) * SteerAngle + (
                        1j * diff_chan))  # output this as a complex number so we can do an x-y plot with the constellation graph
                output_items[3][PhaseStepNumber] = ((1) * SteerAngle + (
                        1j * diff_angle))  # output this as a complex number so we can do an x-y plot with the constellation graph
                PhaseStepNumber = PhaseStepNumber + 1  # increment the phase delta and start this whole again.  This will repeat 140 times
            # print(PhaseStepNumber)
            output_items[0] = output_items[0][0:PhaseStepNumber]
            output_items[1] = output_items[1][0:PhaseStepNumber]
            output_items[2] = output_items[2][0:PhaseStepNumber]
            output_items[3] = output_items[3][0:PhaseStepNumber]
            output_items[4][:] = max_angle  # * (-1)+90

            plt.clf()
            x = output_items[0].real
            y = output_items[0].imag
            plt.plot(x, y)
            plt.draw()
            plt.pause(0.05)
            time.sleep(0.05)

    if x == 1:
        while True:
            # Write code for plotting o/p using single channel
            # We have to step gain over hear. For now it is fixed gain and set in main function
            # Print details such as Tx freq, Rx Freq, bandwidth and Beam Measured/calculated
            # We have to call/set Transreciever in this step. Make it a global variable

            # SteerValues is a sudo line
            # SteerValues = np.arange(-90, 90+steer_res, steer_res)  # convert degrees to radians
            # # Phase delta = 2*Pi*d*sin(theta)/lambda = 2*Pi*d*sin(theta)*f/c
            # PhaseValues = np.degrees(2*3.14159*d*np.sin(np.radians(SteerValues))*SignalFreq/c)

            if res_bits == 1:
                phase_limit = int(225 / phase_step_size) * phase_step_size + phase_step_size
                PhaseValues = np.arange(-phase_limit, phase_limit, phase_step_size)

            gain = []
            angle = []
            max_gain = []
            max_signal = -100000
            max_angle = -90

            for PhDelta in PhaseValues:
                for i in range(0, 1):  # change according to number of adar1000 connected
                    ADAR_set_RxPhase(beam_list[i], device_mode, num_ADARs, PhDelta, phase_step_size)
                    x = beam_list[0].channel2.rx_phase
                    # print(x)

                value1 = (c * np.radians(np.abs(PhDelta))) / (2 * 3.14159 * (SignalFreq) * d) #- sdr.tx_rf_bandwidth * 1000
                clamped_value1 = max(min(1, value1),
                                     -1)  # arcsin argument must be between 1 and -1, or numpy will throw a warning
                theta = np.degrees(np.arcsin(clamped_value1))
                if PhDelta >= 0:
                    SteerAngle = theta  # positive PhaseDelta covers 0deg to 90 deg
                else:
                    SteerAngle = -theta  # negative phase delta covers 0 deg to -90 deg


                total = 0
                for count in range(0, Averages):
                    data_raw = sdr.rx()
                    data = data_raw  # Saving raw data to do calculation for peak value
                    NumSamples = len(data)  # number of samples
                    win = np.blackman(NumSamples)
                    y = data * win
                    sp = np.absolute(np.fft.fft(y))
                    sp = sp[1:-1]
                    sp = np.fft.fftshift(sp)
                    s_mag = np.abs(sp) * 2 / np.sum(
                        win)  # Scale FFT by window and /2 since we are using half the FFT spectrum
                    s_mag = np.maximum(s_mag, 10 ** (-15))
                    s_dbfs = 20 * np.log10(s_mag / (2 ** 12))  # Pluto is a 12 bit ADC, so use that to convert to dBFS
                    total = total + max(s_dbfs)  # sum up all the loops, then we'll average
                PeakValue = total / Averages
                if PeakValue > max_signal:  # for the largest value, save the data so we can plot it in the FFT window
                    max_signal = PeakValue
                    max_angle = SteerAngle
                    data = data_raw
                    NumSamples = len(data)  # number of samples
                    win = np.blackman(NumSamples)
                    y = data * win
                    sp = np.absolute(np.fft.fft(y))
                    sp = sp[1:-1]
                    sp = np.fft.fftshift(sp)
                    s_mag = np.abs(sp) * 2 / np.sum(
                        win)  # Scale FFT by window and /2 since we are using half the FFT spectrum
                    s_mag = np.maximum(s_mag, 10 ** (-15))
                    max_gain = 20 * np.log10(
                        s_mag / (2 ** 12))  # Pluto is a 12 bit ADC, so use that to convert to dBFS
                    ts = 1 / float(sdr.sample_rate)
                    xf = np.fft.fftfreq(NumSamples, ts)
                    xf = np.fft.fftshift(xf[1:-1])  # this is the x axis (freq in Hz) for our fft plot
                gain.append(PeakValue)
                angle.append(SteerAngle)
            ArrayGain = gain
            ArrayAngle = angle
            plt.clf()
            plt.ylim(-50,-10)
            # plt.plot(xf/1e6, max_gain)
            plt.scatter(ArrayAngle, ArrayGain)
            # plt.plot(ArrayAngle, ArrayGain, '-o', ms=5, alpha=0.7, mfc='blue')

            max_gain = max(ArrayGain)
            index_max_gain = np.where(ArrayGain == max_gain)
            index_max_gain = index_max_gain[0]
            max_angle = ArrayAngle[int(index_max_gain[0])]
            plt.axhline(y=max_gain, color='blue', linestyle="--", alpha=0.3)
            plt.axvline(x=max_angle, color = 'red', linestyle=":", alpha=0.3)
            plt.pause(0.05)
            time.sleep(0.05)

            # x = phaser0.channel2.rx_phase
            # print(x)
            # Pluto_Rx(sdr)


def main():

    sw_tx = 1  # Switch to toggle between Pluto and ad9361. Pluto == 1(1r1t) and ad9361 == 2(2r2t)
    # Select and connect the Transreciever
    if sw_tx == 2:
        sdr = adi.ad9361(uri="ip:192.168.2.1")
        ad9361_init(sdr)
    else:
        sdr = adi.Pluto(uri="ip:192.168.2.1")
        PLUTO_init(sdr)
    # if sdr == 'ad9361':
    #     print(sdr)
    time.sleep(1)
    # channel 4 is the third, and channel 3 is the fourth
    # Select and connect the Beamformer/s. Each Beamformer has 4 channels
    # global  adar_0
    adar_0 = adi.adar1000(
        uri="ip:analog.local",
        chip_id="BEAM0",
        array_element_map=[[1, 2, 3, 4]],
        channel_element_map=[2, 1, 4, 3],
    )
    # adar_1 = adi.adar1000(
    #     uri="ip:analog.local",
    #     chip_id="BEAM1",
    #     array_element_map=[[5, 6, 7, 8]],
    #     channel_element_map=[6, 5, 8, 7],
    # )
    # adar_2 = adi.adar1000(
    #     uri="ip:analog.local",
    #     chip_id="BEAM2",
    #     array_element_map=[[9, 10, 11, 12]],
    #     channel_element_map=[10, 9, 12, 11],
    # )
    # adar_3 = adi.adar1000(
    #     uri="ip:analog.local",
    #     chip_id="BEAM3",
    #     array_element_map=[[13, 14, 15, 16]],
    #     channel_element_map=[14, 13, 16, 15],
    # )

    # Initialize the Transmitter of Transreciever
    device_mode = "rx"  # Mode of operation of beamformer. It can be Tx, Rx or Detector
    beam_list = [adar_0]  # adar_1, adar_2, adar_3]  # List of Beamformers in order to steup and configure Individually.

    # initialize the ADAR1000
    x = adar_0.channel2.rx_gain
    print(x)
    for adar in beam_list:
        ADAR_init(adar, device_mode)  # resets the ADAR1000, then reprograms it to the standard config/ Known state

    for adar in beam_list:
        ADAR_set_RxTaper(adar, device_mode)  # Set gain of each channel of all beamformer according to the Cal Values
    x = adar_0.channel2.rx_phase
    print(x)
    # ADAR_set_RxPhase(beam_list[0], device_mode, 1, 45, 2.8125)
    ADAR_Plotter(beam_list, device_mode, sdr)  # Rx down-converted signal and plot it to get sinc pattern


main()
