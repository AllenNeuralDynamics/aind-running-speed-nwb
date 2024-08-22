""" top level run script """

import utils
import pandas as pd
import numpy as np 
import pynwb
import glob

import os
import shutil
from hdmf_zarr import NWBZarrIO

DEFAULT_RUNNING_SPEED_UNITS = {
    "velocity": "cm/s",
    "vin": "V",
    "vsig": "V",
    "rotation": "radians"
}

def extract_running_speeds(
        frame_times, dx_deg, vsig, vin, wheel_radius, subject_position,
        use_median_duration=False
):
    # the first interval does not have a known start time, so we can't compute
    # an average velocity from dx
    dx_rad = utils.degrees_to_radians(dx_deg[1:])

    start_times = frame_times[:-1]
    end_times = frame_times[1:]

    durations = end_times - start_times
    if use_median_duration:
        angular_velocity = dx_rad / np.median(durations)
    else:
        angular_velocity = dx_rad / durations

    radius = wheel_radius * subject_position
    linear_velocity = utils.angular_to_linear_velocity(angular_velocity, radius)

    df = pd.DataFrame(
        {
            "start_time": start_times,
            "end_time": end_times,
            "velocity": linear_velocity,
            "net_rotation": dx_rad,
        }
    )

    # due to an acquisition bug (the buffer of raw orientations may be updated
    # more slowly than it is read, leading to a 0 value for the change in
    # orientation over an interval) there may be exact zeros in the velocity.
    df = df[~(np.isclose(df["net_rotation"], 0.0))]

    return df


def add_running_speed_to_nwbfile(nwbfile, running_speed, units=None):
    if units is None:
        units = DEFAULT_RUNNING_SPEED_UNITS

    running_mod = pynwb.ProcessingModule("running", "running speed data")
    nwbfile.add_processing_module(running_mod)

    running_speed_timeseries = pynwb.base.TimeSeries(
        name="running_speed",
        timestamps=running_speed["start_time"].values,
        data=running_speed["velocity"].values,
        unit=units["velocity"]
    )

    rotation_timeseries = pynwb.base.TimeSeries(
        name="running_wheel_rotation",
        timestamps=running_speed_timeseries,
        data=running_speed["net_rotation"].values,
        unit=units["rotation"]
    )

    running_mod.add_data_interface(running_speed_timeseries)
    running_mod.add_data_interface(rotation_timeseries)

    return nwbfile


def add_raw_running_data_to_nwbfile(nwbfile, raw_running_data, units=None):
    if units is None:
        units = DEFAULT_RUNNING_SPEED_UNITS

    raw_rotation_timeseries = pynwb.base.TimeSeries(
        name="raw_running_wheel_rotation",
        timestamps=np.array(raw_running_data["frame_time"]),
        data=raw_running_data["dx"].values,
        unit=units["rotation"]
    )

    vsig_ts = pynwb.base.TimeSeries(
        name="running_wheel_signal_voltage",
        timestamps=raw_rotation_timeseries,
        data=raw_running_data["vsig"].values,
        unit=units["vsig"]
    )

    vin_ts = pynwb.base.TimeSeries(
        name="running_wheel_supply_voltage",
        timestamps=raw_rotation_timeseries,
        data=raw_running_data["vin"].values,
        unit=units["vin"]
    )

    nwbfile.add_acquisition(raw_rotation_timeseries)
    nwbfile.add_acquisition(vsig_ts)
    nwbfile.add_acquisition(vin_ts)

    return nwbfile

def run():
    """ basic run function """
    # Define the source pattern and destination path
    source_pattern = r'/data/nwb/*.nwb'  # Adjust this pattern as needed
    destination_dir = '/results/nwb/'

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Find all directories matching the source pattern
    source_paths = glob.glob(source_pattern)

    # Copy each matching directory to the destination directory
    for source_path in source_paths:
        destination_path = os.path.join(destination_dir, os.path.basename(source_path))
        print(source_path, destination_path)
        shutil.copytree(source_path, destination_path, dirs_exist_ok=True)

    # Specify the directory you want to search
    base_dir = '/data'

    #pkl_pattern = r'/data/behavior/*.stim.pkl'
    #sync_pattern = r'/data/behavior/*.sync'
    pkl_pattern = r'/data/behavior/*.pkl'
    sync_pattern = r'/data/ophys/*.h5'
    nwb_pattern = r'/results/nwb/*.nwb'

    # Find the matching files using glob
    pkl_files = glob.glob(pkl_pattern)
    sync_files = glob.glob(sync_pattern)
    nwb_files = glob.glob(nwb_pattern)


    print(pkl_files, sync_files, nwb_files)
    # Ensure there's exactly one match for each (or handle as needed)
    if len(pkl_files) == 1 and len(sync_files) == 1 and len(nwb_files) == 1:
        pkl_file = pkl_files[0]
        sync_file = sync_files[0]
        nwb_file = nwb_files[0]
        stim_file = pd.read_pickle(pkl_file)
        sync_dataset = utils.load_sync(sync_file)

        # Why the rising edge? See Sweepstim.update in camstim. This method does:
        # 1. updates the stimuli
        # 2. updates the "items", causing a running speed sample to be acquired
        # 3. sets the vsync line high
        # 4. flips the buffer
        frame_times = utils.get_edges(sync_dataset,
            "rising", ('frames', 'stim_vsync', 'vsync_stim'), units="seconds"
        )


        num_raw_timestamps = len(frame_times)
        print(num_raw_timestamps)
        trimmed_times = utils.trim_discontiguous_times(frame_times)
        print(len(trimmed_times))
        

        dx_deg = utils.running_from_stim_file(stim_file, "dx", num_raw_timestamps)
        if len(dx_deg) > num_raw_timestamps:
            num_raw_timestamps = len(dx_deg)

        if num_raw_timestamps != len(dx_deg):
            raise ValueError(
                f"found {num_raw_timestamps} rising edges on the vsync line, "
                f"but only {len(dx_deg)} rotation samples"
            )

        vsig = utils.running_from_stim_file(stim_file, "vsig", num_raw_timestamps)
        vin = utils.running_from_stim_file(stim_file, "vin", num_raw_timestamps)

        if len(vin) != len(dx_deg):
            vin = np.concatenate((vin, np.zeros((len(dx_deg) - len(vin)))))

        if len(vsig) != len(dx_deg):
            vsig = np.concatenate((vsig, np.zeros((len(dx_deg) - len(vsig)))))

        velocities = extract_running_speeds(
            frame_times=frame_times,
            dx_deg=dx_deg,
            vsig=vsig,
            vin=vin,
            wheel_radius=8.255,
            subject_position=2/3,
            use_median_duration=True
        )

        raw_data = pd.DataFrame(
            {"vsig": vsig, "vin": vin, "frame_time": frame_times, "dx": dx_deg}
        )
        print(raw_data)
        print(velocities)

        io = NWBZarrIO(nwb_file, "r+", load_namespaces=True)
        input_nwb = io.read()
        input_nwb = add_running_speed_to_nwbfile(input_nwb, velocities)
        input_nwb = add_raw_running_data_to_nwbfile(input_nwb, raw_data)
        io.write(input_nwb)
        io.close()
    

if __name__ == "__main__": run()