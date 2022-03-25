# --------------------LICENSE AGREEMENT----------------------------------------
# Copyright (c) 2020 Analog Devices, Inc.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#   - Modified versions of the software must be conspicuously marked as such.
#   - This software is licensed solely and exclusively for use with
#   processors/products manufactured by or for Analog Devices, Inc.
#   - This software may not be combined or merged with other code in any manner
#   that would cause the software to become subject to terms and conditions
#   which differ from those listed here.
#   - Neither the name of Analog Devices, Inc. nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#   - The use of this software may or may not infringe the patent rights of
#   one or more patent holders.  This license does not release you from the
#   requirement that you obtain separate licenses from these patent holders
#   to use this software.
#
# THIS SOFTWARE IS PROVIDED BY ANALOG DEVICES, INC. AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# NON-INFRINGEMENT, TITLE, MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL ANALOG DEVICES, INC. OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, PUNITIVE OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# DAMAGES ARISING OUT OF CLAIMS OF INTELLECTUAL PROPERTY RIGHTS INFRINGEMENT;
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# 2020-02-24-7CBSD SLA
# -----------------------------------------------------------------------------


import sys
import matplotlib.pyplot as plt
import numpy as np
import sin_params as sp
import time
import queue
import threading
import csv

""" Using local copy of pyadi as module as master pyadi is inform of .egg """
sys.path.insert(0,
                r"C:\Users\DTrivedi\OneDrive - Analog Devices, Inc\Desktop\Work_database\Rotations\Precision Converter Team\ad4630_python\ad4630-24_python_files\pyadi-iio")
import adi


"""Other parameters to configure data"""
device_name = "ad4630-24"
fs = 2000000  # Sampling Frequency
N = 2 ** 10  # Length of rx buffer
get_data_for_secs = 5
data_collection = "plot_data"  # you can either plot_data or log_data


class ad463x_plot:
    def __init__(self):
        """ Instantiate the device and set th parameters."""
        self.t1 = threading.Thread(target=self.get)
        self.t2 = threading.Thread(target=self.plot)
        self.t3 = threading.Thread(target=self.log)

        self.adc = adi.ad4630(uri="ip:analog.local",
                              device_name=device_name)  # To connect via ip address 169.254.92.202
        self.adc.rx_buffer_size = N
        self.adc.sample_rate = fs

        """To switch the device b/w low_power_mode and normal_operating_mode."""
        self.adc.operating_mode = "normal_operating_mode"

        """ Prints Current output data mode"""
        print(self.adc.output_data_mode)

        """sample_averaging is only supported by 30bit mode. and in this mode it cannot be OFF."""
        if self.adc.output_data_mode == "30bit_avg":
            self.adc.sample_averaging = 64

        """ Differential Channel attributes"""
        if self.adc.output_data_mode != "32bit_test_pattern":
            self.adc.chan0.hw_gain = 1
            self.adc.chan0.offset = 0
            if device_name == "ad4630-24":
                self.adc.chan1.hw_gain = 1
                self.adc.chan1.offset = 0
        self.q = queue.Queue(maxsize=0)
        self.data_getter()

    def data_getter(self):
        """Function to start data collection thread and plot/log it accordingly"""

        self.t1.start()
        if data_collection == "plot_data":
            time.sleep(5)  # start plotting data after 5 Seconds
            self.t2.start()

        elif data_collection == "log_data":
            time.sleep(1)
            self.t3.start()

    def get(self):
        """Continuously Sample data and stuff it in a queue."""

        t_end = time.time() + get_data_for_secs
        while time.time() < t_end:
            data = self.adc.rx()  # Recieve the data
            for ch in range(0, len(data)):
                self.q.put(data[ch])

    def plot(self):
        """Plot the data in queue after certain time if data collection method is plot"""
        x = np.arange(0, N)
        while not self.q.empty():
            plt.ion()
            for ch in range(len(self.adc._ctrl.channels)):
                wf_data = self.q.get()
                plt.figure(self.adc._ctrl.channels[ch]._name)  # Using hidden functions in example code is not adviced
                plt.clf()
                plt.step(x * (1000 / fs), wf_data)
            plt.draw()
            plt.pause(10)

    def log(self):
        """Log the data to a csv file if data collection method is log"""

        with open('data.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            if len(self.adc._ctrl.channels) == 2:
                csv_writer.writerow(("differential0", "differential1"))
            else:
                csv_writer.writerow(("differential0", "common0", "differential1", "common1"))
        while not self.q.empty():
            if len(self.adc._ctrl.channels) == 2:
                differential0 = self.q.get()
                differential1 = self.q.get()
            else:
                differential0 = self.q.get()
                common0 = self.q.get()
                differential1 = self.q.get()
                common1 = self.q.get()

            with open('data.csv', 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                row_count = 0
                for row in differential0:
                    if len(self.adc._ctrl.channels) == 2:
                        csv_writer.writerow((differential0[row_count], differential1[row_count]))
                    else:
                        csv_writer.writerow((differential0[row_count], common0[row_count], differential1[row_count], common1[row_count]))
                    row_count += 1
        csv_file.close()


if __name__ == '__main__':
    app = ad463x_plot()
