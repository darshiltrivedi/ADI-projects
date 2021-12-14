from adi.ad936x import ad9361, Pluto


class sdr(ad9361):
    def __init__(self, uri):
        ad9361.__init__(self, uri)

    # This method configures sdr w.r.t to current freq plan
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
