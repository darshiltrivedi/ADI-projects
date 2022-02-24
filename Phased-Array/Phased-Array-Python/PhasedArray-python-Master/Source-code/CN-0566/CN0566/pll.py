from adi.adf4159 import adf4159


class pll(adf4159):
    def __init__(self, uri):
        adf4159.__init__(self, uri)

    # This method configures pll w.r.t to current freq plan
    def configure(self):
        self.frequency = 6247500000
        self.freq_dev_step = 5690
        self.freq_dev_range = 0
        self.freq_dev_time = 0
        self.powerdown = 0
        self.ramp_mode = "disabled"

