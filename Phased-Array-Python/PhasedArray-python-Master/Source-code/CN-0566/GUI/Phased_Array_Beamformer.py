import sys
import adi
import numpy as np
import time
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

# This is a seprate Pluto whose tx is connected back to Rx and will be used in session 2.
# For Now, while I am testing I made this global as I have a seprate pluto on my work bench connected to RPI
pluto_ss2 = adi.Pluto(uri="ip:192.168.2.1")  # Pluto on RPI doesn't work with pyadi

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('Phased_Array_beamformer.ui', self)  # Load the User Interface file
        self.start_button.clicked.connect(self.start_plot)  # call start_plot method when button clicked
        self.stop_button.clicked.connect(self.stop_plot)  # call stop_plot method when button clicked
        self.calibrate_button.clicked.connect(self.calibration)  # call calibration method when button clicked
        self.show()


    def start_plot(self):
        # This function is connected to start plot button and it plots based on current tab and all tab values
        # The SDR tab in configuration area and SDR FFT tab in plot area should work seprately. So only when you have this tabs you can see fft plot i.e. of session 2
        # All other tabs in configuration area are the input values and based on that values we plot according to current plot tab.
        # Try to add a status bar indicating current status or messages i.e. Switch to appropriate Tab etc.

        if (self.displaytab.currentIndex() == 3 and self.configarea.currentIndex() == 0):
            # Plot FFT for session 2
            self.fft_worker = FFT(self.sdr_test1_val, self.sdr_test2_val, self.sdr_bw_val, self.sdr_tx_val, self.sdr_rx_val, self.sdr_sr_val, self.filter_box, self.fft_sdr_display)
            self.fft_worker.start()

        elif (self.displaytab.currentIndex() != 3 and self.configarea.currentIndex() == 0):
            # Display error message if Not in FFT_S2 display area
            print("Switch to appropriate Tab")

        elif (self.displaytab.currentIndex() == 3 and self.configarea.currentIndex() != 0):
            # Display error message if Not in FFT_S2 Config area
            print("Switch to appropriate Tab")

        elif (self.displaytab.currentIndex() == 0 and self.configarea.currentIndex() != 0):
            # Currently I am working on this. Update it to git as I complete testing of it.
            if self.adf4159_radio.isChecked():
                self.pll = adi.adf4159(uri=str(self.rpi_ip_val.toPlainText()))
                if self.oneadar_radio.isChecked():
                    self.sdr = adi.Pluto(uri=str(self.LO_ip_val.toPlainText())).rx
                elif self.twoadar_radio.isChecked():
                    self.sdr = adi.ad9361(uri=str(self.LO_ip_val.toPlainText())).rx

            elif self.ad9361_radio.isChecked():
                if self.oneadar_radio.isChecked():
                    self.sdr = adi.Pluto(uri=str(self.LO_ip_val.toPlainText())).rx
                    self.pll = adi.Pluto(uri=str(self.LO_ip_val.toPlainText())).tx
                elif self.twoadar_radio.isChecked():
                    self.sdr = adi.ad9361(uri=str(self.LO_ip_val.toPlainText())).rx
                    self.pll = adi.ad9361(uri=str(self.LO_ip_val.toPlainText())).tx

            self.rplot_worker = Rect_plot(self.sdr, self.pll)
            self.rplot_worker.start()

            print("Rectangular Plot")

        elif (self.displaytab.currentIndex() == 1 and self.configarea.currentIndex() != 0):
            print("Polar Plot")

        elif (self.displaytab.currentIndex() == 2 and self.configarea.currentIndex() != 0):
            print("FFT at Peak Angle")

    def stop_plot(self):
        # This function is connected to stop plot button. It stops plotting all the plot if any plotting currently
        if self.fft_worker.isRunning():
            self.fft_worker.terminate()
        if self.rplot_worker.isRunning():
            self.rplot_worker.terminate()

    def calibration(self):
        # This function is connected to calibrate button and it does calibration based on radio button selected.
        if self.gain_cal_radio.isChecked():
            # add gain Calibration Routine here
            # place this routine in a seprate thread as it takes time to complete
            print("Gain Calibration is going on")

        elif self.phase_cal_radio.isChecked():
            # add Phase Calibration Routine here
            # place this routine in a seprate thread as it takes time to complete
            print("Phase Calibration is going on")

        elif self.full_cal_radio.isChecked():
            # add both Calibration Routine here
            # place this routine in a seprate thread as it takes time to complete
            print("Full System Calibration is going on")

class FFT(QThread):
    # I could not inherit from main thread to worker thread maybe pyqt5 does not permit it as value cannot be accessed at same time
    # So I Pass those value/attr (from Main Thread) and instantiate them in the constructor of worker thread
    def __init__(self, sdr_test1_val, sdr_test2_val, sdr_bw_val, sdr_tx_val, sdr_rx_val, sdr_sr_val, filter_box, fft_sdr_display, parent=None):
        super(FFT, self).__init__(parent)
        self.sdr_test1_val = sdr_test1_val
        self.sdr_test2_val = sdr_test2_val
        self.sdr_bw_val = sdr_bw_val
        self.sdr_tx_val = sdr_tx_val
        self.sdr_rx_val = sdr_rx_val
        self.sdr_sr_val = sdr_sr_val
        self.filter_box = filter_box
        self.fft_sdr_display = fft_sdr_display

    def run(self):
        # Setup Frequency and Test Tones
        freq_band = 400000
        tt1_freq = int(self.sdr_test1_val.toPlainText()) * 100000
        tt2_freq = int(self.sdr_test2_val.toPlainText()) * 100000
        buf_len = 2 ** 13
        # center_freq = np.array([])
        # Setup Pluto
        pluto_ss2._tx_buffer_size = 2 ** 13
        pluto_ss2.rx_buffer_size = 2 ** 13
        pluto_ss2.tx_rf_bandwidth = int(self.sdr_bw_val.toPlainText()) * 1000000
        pluto_ss2.tx_lo = int(self.sdr_tx_val.toPlainText()) * 1000000
        pluto_ss2.rx_lo = int(self.sdr_rx_val.toPlainText()) * 1000000
        pluto_ss2.tx_cyclic_buffer = True
        pluto_ss2.tx_hardwaregain = -30
        pluto_ss2.rx_hardwaregain_chan0 = 80
        pluto_ss2.gain_control_mode = 'slow_attack'
        pluto_ss2.sample_rate = int(self.sdr_sr_val.toPlainText())
        pluto_ss2.dds_scales = ('0.1', '0.25', '0', '0', '0.1', '0.25', '0', '0')
        pluto_ss2.dds_enabled = (1, 1, 1, 1, 1, 1, 1, 1)
        if self.filter_box.isChecked() == True:
            pluto_ss2.filter = "LTE20_MHz.ftr"
        time.sleep(0.2)
        window = np.hamming(buf_len)
        avg_step = 0
        fft_rxvals_iq = np.array([])
        fftfreq_rxvals_iq = np.array([])
        avgband = np.array([])
        while True:
            for i in range(0, 1):
                pluto_ss2.dds_frequencies = (str(tt1_freq), str(tt2_freq), str(tt1_freq), str(tt2_freq), 0, 0, 0, 0)
                data = pluto_ss2.rx()
                data_r = window * data.real
                data_i = window * data.imag
                data_complex = data_r + (1j * data_i)
                avgband = np.abs(np.fft.fft(data_complex))
                freq = (np.fft.fftfreq(int(buf_len), 1 / int(pluto_ss2.rx_lo)) + (
                        pluto_ss2.rx_lo + int(freq_band) * i)) / 1e6
                tuple1 = zip(freq, avgband)
                tuple1 = sorted(tuple1, key=lambda x: x[0])
                iq = np.array([nr[1] for nr in tuple1])

                if avg_step == 0:
                    fft_rxvals_iq = np.append(fft_rxvals_iq, iq)
                    fftfreq_rxvals_iq = np.append(fftfreq_rxvals_iq, [nr[0] for nr in tuple1])
                else:
                    # replace older amplitudes per band with newer amplitudes
                    iq = iq * 1 / avg_step + fft_rxvals_iq * (avg_step - 1) / avg_step
                    fft_rxvals_iq = iq

                fft_rxvals_iq_db = 20 * np.log10(fft_rxvals_iq * 2 / (2 ** 11 * buf_len))
                self.fft_sdr_display.axes.plot(fftfreq_rxvals_iq, fft_rxvals_iq_db)
                self.fft_sdr_display.axes.set_xlabel('Frequency in MHz')
                self.fft_sdr_display.axes.set_ylabel('Gain in dB')
                self.fft_sdr_display.canvas.draw()
                avg_step += 1
            # print(fftfreq_rxvals_iq[np.argmax(fft_rxvals_iq_db)])
            window = np.hamming(buf_len)
            avg_step  = 0
            fft_rxvals_iq = np.array([])
            fftfreq_rxvals_iq = np.array([])
            avgband = np.array([])
            self.fft_sdr_display.axes.clear()
            time.sleep(0.1)


# This is an empty worker thread but will connected to rectangular plot display widget.
class Rect_plot(QThread):
    def __init__(self, sdr, pll, parent=None):
        super(Rect_plot, self).__init__(parent)
        self.sdr = sdr
        self.pll = pll

    def run(self):
        pass

# This is an empty worker thread but will connected to Polar plot display widget.
class Polar_plot(QThread):
    def __init__(self, parent=None):
        super(Polar_plot, self).__init__(parent)

    def run(self):
        pass

# This is an empty worker thread but will connected to FFT at peak angle plot display widget.
class FFT_PSA(QThread):
    def __init__(self, parent=None):
        super(FFT_PSA, self).__init__(parent)

    def run(self):
        pass

# This is an empty worker thread but will connected to Calibration routine.
class Calibrate(QThread):
    def __init__(self, parent=None):
        super(Calibrate, self).__init__(parent)

    def run(self):
        pass



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())
