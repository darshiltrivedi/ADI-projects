import sys
import adi
import numpy as np
import time
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

pluto_tx = adi.Pluto(uri="ip:192.168.2.1")  # Pluto on RPI doesn't work with pyadi

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('Phased_Array_beamformer.ui', self)
        self.start_button.clicked.connect(self.start_plot)
        self.stop_button.clicked.connect(self.stop_plot)
        self.show()


    def start_plot(self):
        if (self.displaytab.currentIndex() == 3 and self.configarea.currentIndex() == 0):
            self.fft_worker = FFT(self.sdr_test1_val, self.sdr_test2_val, self.sdr_bw_val, self.sdr_tx_val, self.sdr_rx_val, self.sdr_sr_val, self.filter_box, self.fft_sdr_display)
            self.fft_worker.start()

        elif (self.displaytab.currentIndex() != 3 and self.configarea.currentIndex() == 0):
            print("Switch to appropriate Tab")

        elif (self.displaytab.currentIndex() == 3 and self.configarea.currentIndex() != 0):
            print("Switch to appropriate Tab")

        elif (self.displaytab.currentIndex() == 0 and self.configarea.currentIndex() != 0):
            print("Rectangular Plot")

        elif (self.displaytab.currentIndex() == 1 and self.configarea.currentIndex() != 0):
            print("Polar Plot")

        elif (self.displaytab.currentIndex() == 2 and self.configarea.currentIndex() != 0):
            print("FFT at Peak Angle")

    def stop_plot(self):
        self.fft_worker.terminate()


class FFT(QThread):

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
        pluto_tx._tx_buffer_size = 2 ** 13
        pluto_tx.rx_buffer_size = 2 ** 13
        pluto_tx.tx_rf_bandwidth = int(self.sdr_bw_val.toPlainText()) * 1000000
        pluto_tx.tx_lo = int(self.sdr_tx_val.toPlainText()) * 1000000
        pluto_tx.rx_lo = int(self.sdr_rx_val.toPlainText()) * 1000000
        pluto_tx.tx_cyclic_buffer = True
        pluto_tx.tx_hardwaregain = -30
        pluto_tx.rx_hardwaregain_chan0 = 80
        pluto_tx.gain_control_mode = 'slow_attack'
        pluto_tx.sample_rate = int(self.sdr_sr_val.toPlainText())
        pluto_tx.dds_scales = ('0.1', '0.25', '0', '0', '0.1', '0.25', '0', '0')
        pluto_tx.dds_enabled = (1, 1, 1, 1, 1, 1, 1, 1)
        if self.filter_box.isChecked() == True:
            pluto_tx.filter = "LTE20_MHz.ftr"
        time.sleep(0.2)
        window = np.hamming(buf_len)
        avg_step = 0
        fft_rxvals_iq = np.array([])
        fftfreq_rxvals_iq = np.array([])
        avgband = np.array([])
        while True:
            for i in range(0, 1):
                pluto_tx.dds_frequencies = (str(tt1_freq), str(tt2_freq), str(tt1_freq), str(tt2_freq), 0, 0, 0, 0)
                data = pluto_tx.rx()
                data_r = window * data.real
                data_i = window * data.imag
                data_complex = data_r + (1j * data_i)
                avgband = np.abs(np.fft.fft(data_complex))
                freq = (np.fft.fftfreq(int(buf_len), 1 / int(pluto_tx.rx_lo)) + (
                        pluto_tx.rx_lo + int(freq_band) * i)) / 1e6
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


class Rect_plot(QThread):
    def __init__(self, parent=None):
        super(Rect_plot, self).__init__(parent)

    def run(self):
        pass


class Polar_plot(QThread):
    def __init__(self, parent=None):
        super(Polar_plot, self).__init__(parent)

    def run(self):
        pass


class FFT_PSA(QThread):
    def __init__(self, parent=None):
        super(FFT_PSA, self).__init__(parent)

    def run(self):
        pass


class Calibrate(QThread):
    def __init__(self, parent=None):
        super(Calibrate, self).__init__(parent)

    def run(self):
        pass



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())
