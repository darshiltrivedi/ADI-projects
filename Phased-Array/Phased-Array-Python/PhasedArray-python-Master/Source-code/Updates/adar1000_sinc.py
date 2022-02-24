import time

import adi
from functions import *
from project_config import *


def main():
    # This Main function would be eventually function of GUI
    # all other things will be called inside the GUI
    # Select and connect the Transreciever
    if "adf4159" in tx_source:
        pll = adi.adf4159()
        adf4159_init(pll)
        sdr = adi.ad9361(uri=lo_ip)
        ad9361_init(sdr)

    elif "pluto" in tx_source:
        if sw_tx == 2:
            sdr = adi.ad9361(uri=lo_ip)
            ad9361_init(sdr)
        else:
            sdr = adi.Pluto(uri=lo_ip)
            PLUTO_init(sdr)

    else:
        raise Exception("Please select appropriate transmitter source. Valid options 'pluto' or 'adf4159'")

    # channel 4 is the third, and channel 3 is the fourth
    # Select and connect the Beamformer/s. Each Beamformer has 4 channels
    # global  adar_0
    # Try to make a beam list function in which you can select number of ADAR1000 and the Local Osc that you want
    adar = adi.adar1000_array(
        uri="ip:10.84.10.66",
        chip_ids=["BEAM0", "BEAM1"],
        device_map=[[1, 2]],
        element_map=[[1, 2, 3, 4], [5, 6, 7, 8]],
        device_element_map={
            1: [2, 1, 4, 3],
            2: [6, 5, 8, 7],
        },
    )

#     adar_0 = adi.adar1000(
#         uri="ip:10.84.10.64",
#         chip_id="BEAM0",
#         array_element_map=[[1, 2, 3, 4]],
#         channel_element_map=[2, 1, 4, 3],
#     )
#     adar_1 = adi.adar1000(
#         uri="ip:10.84.10.64",
#         chip_id="BEAM1",
#         array_element_map=[[5, 6, 7, 8]],
#         channel_element_map=[6, 5, 8, 7],
#     )
    # adar_2 = adi.adar1000(
    #     uri="ip:phaser-host.local",
    #     chip_id="BEAM2",
    #     array_element_map=[[9, 10, 11, 12]],
    #     channel_element_map=[10, 9, 12, 11],
    # )
    # adar_3 = adi.adar1000(
    #     uri="ip:phaser-host.local",
    #     chip_id="BEAM3",
    #     array_element_map=[[13, 14, 15, 16]],
    #     channel_element_map=[14, 13, 16, 15],
    # )

    beam_list = []  # List of Beamformers in order to steup and configure Individually.
    for device in adar.devices.values():
        beam_list.append(device)

    # Initialize the Transmitter of Transreciever
#     beam_list = [adar_0, adar_1]  # List of Beamformers in order to steup and configure Individually.

    # initialize the ADAR1000
    for adar in beam_list:
        ADAR_init(adar)  # resets the ADAR1000, then reprograms it to the standard config/ Known state
        ADAR_set_RxTaper(adar)  # Set gain of each channel of all beamformer according to the Cal Values

    ADAR_Plotter(beam_list, sdr)  # Rx down-converted signal and plot it to get sinc pattern


main()
