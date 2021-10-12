import adi
import numpy as np
import matplotlib.pyplot as plt
import time


def PLUTO_init(pluto_tx):
    # Setup Pluto
    pluto_tx._tx_buffer_size = 2 ** 13
    pluto_tx.rx_buffer_size = 2 ** 13
    pluto_tx.tx_rf_bandwidth = 55000000
    pluto_tx.tx_lo = 6000000000
    pluto_tx.rx_lo = 4492000000
    pluto_tx.tx_cyclic_buffer = True
    pluto_tx.tx_hardwaregain_chan0 = -80
    # pluto_tx.rx_hardwaregain_chan0 = 80
    pluto_tx.gain_control_mode_chan0 = 'slow_attack'
    pluto_tx.sample_rate = 3044000
    pluto_tx.dds_scales = ('0.3', '0.25', '0', '0')
    pluto_tx.dds_enabled = (1, 1, 1, 1)
    pluto_tx.filter = "LTE20_MHz.ftr"
    # pluto_tx._rxadc.set_kernel_buffers_count(
    #     1)  # Default is 4 Rx buffers are stored, but we want to change and immediately measure the result, so buffers=1
    # rx = pluto_tx._ctrl.find_channel('voltage0')
    # rx.attrs['quadrature_tracking_en'].value = '1'  # set to '1' to enable quadrature tracking
    # pluto_tx.rx_buffer_size = int(4 * 256)
    # pluto_tx.tx_buffer_size = int(2 ** 18)
    # pluto_tx.tx_hardwaregain_chan0 = -80  # this is a negative number between 0 and -88
    # pluto_tx.gain_control_mode_chan0 = "manual"  # We must be in manual gain control mode (otherwise we won't see the peaks and nulls!)
    # pluto_tx.rx_hardwaregain_chan0 = int(Rx_gain)
    # pluto_tx.dds_enabled = [1, 1, 1, 1]                  #DDS generator enable state
    # pluto_tx.dds_frequencies = [0.1e6, 0.1e6, 0.1e6, 0.1e6]      #Frequencies of DDSs in Hz
    # pluto_tx.dds_scales = [1, 1, 0, 0]                   #Scale of DDS signal generators Ranges [0,1]
    # pluto_tx.dds_single_tone(int(0.0e6), 0.9, 0)    # pluto_tx.dds_single_tone(tone_freq_hz, tone_scale_0to1, tx_channel)

    time.sleep(0.1)
    # Pluto_Rx(pluto_tx)


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


def ADAR_init(phaser, adar_mode):
    # Configure ADAR1000
    # phaser.initialize_devices()  # Always Intialize the device 1st as reset is performed at Initialization
    phaser.reset()  # Performs a soft reset of the device (writes 0x81 to reg 0x00)
    phaser._ctrl.reg_write(0x400, 0x55)  # This trims the LDO value to approx. 1.8V (to the center of its range)

    phaser.sequencer_enable = False
    phaser.beam_mem_enable = False  # RAM control vs SPI control of the beam state, reg 0x38, bit 6.  False sets bit high and SPI control
    phaser.bias_mem_enable = False  # RAM control vs SPI control of the bias state, reg 0x38, bit 5.  False sets bit high and SPI control
    phaser.pol_state = False  # Polarity switch state, reg 0x31, bit 0. True outputs -5V, False outputs 0V
    phaser.pol_switch_enable = False  # Enables switch driver for ADTR1107 switch, reg 0x31, bit 3
    phaser.tr_source = 'spi'  # TR source for chip, reg 0x31 bit 2.  'external' sets bit high, 'spi' sets bit low
    phaser.tr_spi = 'rx'  # TR SPI control, reg 0x31 bit 1.  'tx' sets bit high, 'rx' sets bit low
    phaser.tr_switch_enable = True  # Switch driver for external switch, reg0x31, bit 4
    phaser.external_tr_polarity = True  # Sets polarity of TR switch compared to TR state of ADAR1000.  True outputs 0V in Rx mode

    phaser.rx_vga_enable = True  # Enables Rx VGA, reg 0x2E, bit 0.
    phaser.rx_vm_enable = True  # Enables Rx VGA, reg 0x2E, bit 1.
    phaser.rx_lna_enable = True  # Enables Rx LNA, reg 0x2E, bit 2.
    phaser.rx_lna_bias_current = 8  # Sets the LNA bias to the middle of its range
    phaser.rx_vga_vm_bias_current = 22  # Sets the VGA and vector modulator bias.  I thought this should be 22d, but Stingray has it as 85d????

    phaser.tx_vga_enable = True  # Enables Tx VGA, reg 0x2F, bit0
    phaser.tx_vm_enable = True  # Enables Tx Vector Modulator, reg 0x2F, bit1
    phaser.tx_pa_enable = True  # Enables Tx channel drivers, reg 0x2F, bit2
    phaser.tx_pa_bias_current = 6  # Sets Tx driver bias current
    phaser.tx_vga_vm_bias_current = 22  # Sets Tx VGA and VM bias.  I thought this should be 22d, but stingray has as 45d??????
    # print("Done")

    if adar_mode == "rx":
        # Configure the device for Rx mode
        phaser.mode = "rx"  # Mode of operation, bit 5 of reg 0x31. "rx", "tx", or "disabled"
        SELF_BIASED_LNAs = True
        if SELF_BIASED_LNAs:
            # Allow the external LNAs to self-bias
            phaser.lna_bias_out_enable = False  # this writes 0xA0 0x30 0x00.  What does this do? # Disabling it allows LNAs to stay in self bias mode all the time
            # phaser._ctrl.reg_write(0x30, 0x00)   #Disables PA and DAC bias
        else:
            # Set the external LNA bias
            phaser.lna_bias_on = -0.7  # this writes 0x25 to register 0x2D.  This is correct.  But oddly enough, it doesn't first write 0x18 to reg 0x00....
            # phaser._ctrl.reg_write(0x30, 0x20)   #Enables PA and DAC bias.  I think this would be needed too?

        # Enable the Rx path for each channel
        for channel in phaser.channels:
            channel.rx_enable = True  # this writes reg0x2E with data 0x00, then reg0x2E with data 0x20.  So it overwrites 0x2E, and enables only one channel
            # phaser._ctrl.reg_write(0x2E, 0x7F)    #Enables all four Rx channels, the Rx LNA, Rx Vector Modulator and Rx VGA

    # Configure the device for Tx mode
    else:
        phaser.mode = "tx"

        # Enable the Tx path for each channel and set the external PA bias
        for channel in phaser.channels:
            channel.tx_enable = True
            channel.pa_bias_on = -2

    if adar_mode == "rx":
        phaser.latch_rx_settings()  # writes 0x01 to reg 0x28.
    else:
        phaser.latch_tx_settings()  # writes 0x02 to reg 0x28.


def ADAR_set_RxTaper(phaser, adar_mode):
    # Set the array phases to 10°, 20°, 30°, and 40° and the gains to max gain (0x7f, or 127)
    for channel in phaser.channels:
        # print(channel)
        # Set the gain and phase depending on the device mode
        if adar_mode == "rx":
            # channel.rx_phase = 90 #channel.array_element_number  * 10  # writes to I and Q registers.
            channel.rx_gain = 127
            # print(channel.rx_phase)
        else:
            # channel.tx_phase = channel.array_element_number * 10  # writes to I and Q registers.
            channel.tx_gain = 127

    # Latch in the new gains & phases
    if adar_mode == "rx":
        phaser.latch_rx_settings()  # writes 0x01 to reg 0x28.
    else:
        phaser.latch_tx_settings()  # writes 0x02 to reg 0x28.


def ADAR_set_RxPhase(phaser, adar_mode, ADAR_ADD, Ph_Diff, ph_step_size):
    Rx1_Phase_Cal = 0
    Rx2_Phase_Cal = 0
    Rx3_Phase_Cal = 0
    Rx4_Phase_Cal = 0
    Rx5_Phase_Cal = 0
    Rx6_Phase_Cal = 0
    Rx7_Phase_Cal = 0
    Rx8_Phase_Cal = 0

    if ADAR_ADD == 1:
        Phase_A = ((np.rint(
            Ph_Diff * 0 / ph_step_size) * ph_step_size) + Rx1_Phase_Cal) % 360  # round each value to the nearest step size increment
        Phase_B = ((np.rint(Ph_Diff * 1 / ph_step_size) * ph_step_size) + Rx2_Phase_Cal) % 360
        Phase_C = ((np.rint(Ph_Diff * 2 / ph_step_size) * ph_step_size) + Rx3_Phase_Cal) % 360
        Phase_D = ((np.rint(Ph_Diff * 3 / ph_step_size) * ph_step_size) + Rx4_Phase_Cal) % 360

    elif ADAR_ADD == 2:
        Phase_A = ((np.rint(Ph_Diff * 4 / ph_step_size) * ph_step_size) + Rx5_Phase_Cal) % 360
        Phase_B = ((np.rint(Ph_Diff * 5 / ph_step_size) * ph_step_size) + Rx6_Phase_Cal) % 360
        Phase_C = ((np.rint(Ph_Diff * 6 / ph_step_size) * ph_step_size) + Rx7_Phase_Cal) % 360
        Phase_D = ((np.rint(Ph_Diff * 7 / ph_step_size) * ph_step_size) + Rx8_Phase_Cal) % 360
    channel_phase_value = [Phase_A, Phase_B, Phase_C, Phase_D]
    # print(channel_phase_value)

    i = 0
    for channel in phaser.channels:
        # Set phase depending on the device mode
        if adar_mode == "rx":
            channel.rx_phase = channel_phase_value[
                i]  # writes to I and Q registers values according to Table 13-16 from datasheet.
            phaser.latch_rx_settings()
        i = i + 1

    return None


def ADAR_Plotter(channel_list, device_mode,sdr):
    phase_step_size = 2.8125
    PhaseValues = np.arange(-196.875, 196.875,
                            phase_step_size)  # These are all the phase deltas (i.e. phase difference between Rx1 and Rx2, then Rx2 and Rx3, etc.) we'll sweep.
    PhaseStepNumber = 0  # this is the number of phase steps we'll take (140 in total).  At each phase step, we set the individual phases of each of the Rx channels
    max_signal = -100  # Reset max_signal.  We'll keep track of the maximum signal we get as we do this 140 loop.
    max_angle = 0  # Reset max_angle.  This is the angle where we saw the max signal.  This is where our compass will point.
    c = 299792458  # speed of light in m/s
    d = 0.015  # element to element spacing of the antenna
    SignalFreq = 10525000000
    num_ADARs = 1  # Ask Jon difference between ADDR1 and ADDR2
    Averages = 1
    output_items = np.zeros((5, 140))

    for PhDelta in PhaseValues:
        for i in range(0, 3):
            ADAR_set_RxPhase(channel_list[i], device_mode, num_ADARs, PhDelta, phase_step_size)

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
            chan1 = data[0]  # Rx1 data
            chan2 = data[1]  # Rx2 data
            sum_chan = chan1 + chan2
            delta_chan = chan1 - chan2
            N = len(data)  # number of samples
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

        # print(PeakValue_sum, PeakValue_angle, PeakValue_delta)
        # plt.clf()
        # plt.plot(PeakValue_sum)
        # plt.draw()
        # plt.pause(0.05)
        # time.sleep(0.05)
        # PLUTO_init(sdr)

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
    plt.show()
    plt.pause(0.05)
    time.sleep(0.05)
    Pluto_Rx(sdr)


def main():
    # Select and connect the Transreciever
    sdr = adi.Pluto(uri="ip:192.168.2.2")

    # channel 4 is the third, and channel 3 is the fourth
    # Select and connect the Beamformer/s. Each Beamformer has 4 channels
    phaser0 = adi.adar1000(
        uri="ip:analog.local",
        chip_id="BEAM0",
        array_element_map=[[1, 2, 3, 4]],
        channel_element_map=[2, 1, 4, 3],
    )
    phaser1 = adi.adar1000(
        uri="ip:analog.local",
        chip_id="BEAM1",
        array_element_map=[[5, 6, 7, 8]],
        channel_element_map=[6, 5, 8, 7],
    )
    phaser2 = adi.adar1000(
        uri="ip:analog.local",
        chip_id="BEAM2",
        array_element_map=[[9, 10, 11, 12]],
        channel_element_map=[10, 9, 12, 11],
    )
    phaser3 = adi.adar1000(
        uri="ip:analog.local",
        chip_id="BEAM3",
        array_element_map=[[13, 14, 15, 16]],
        channel_element_map=[14, 13, 16, 15],
    )

    # Initialize the Transmitter of Transreciever
    PLUTO_init(sdr)

    device_mode = "rx"  # Mode of operation of beamformer. It can be Tx, Rx or Detector
    channel_list = [phaser0, phaser1, phaser2,
                    phaser3]  # List of Beamformers in order to steup and configure Individually.

    # initialize the ADAR1000
    for phase_channel in channel_list:
        ADAR_init(phase_channel,
                  device_mode)  # resets the ADAR1000, then reprograms it to the standard config/ Known state

    for phase_channel in channel_list:
        ADAR_set_RxTaper(phase_channel,
                         device_mode)  # Set gain of each channel of all beamformer according to the Cal Values

    ADAR_Plotter(channel_list, device_mode, sdr)


main()
