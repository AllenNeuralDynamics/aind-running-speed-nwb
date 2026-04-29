from hdmf_zarr import NWBZarrIO as nwbio
import matplotlib.pyplot as plt
import numpy as np

nwb_path = "/root/capsule/results/multiplane-ophys_837568_2026-03-06_13-39-00.nwb"

io = nwbio(nwb_path,mode='r')
nwb = io.read()
running = nwb.processing['running']['running_speed']
r_data = np.array(running.data)
r_timestamps = np.array(running.timestamps)
plt.plot(r_timestamps,r_data)
plt.savefig('/root/capsule/results/running.png')