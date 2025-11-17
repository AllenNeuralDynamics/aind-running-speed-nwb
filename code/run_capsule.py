"""top level run script"""

import argparse
import logging
import os
import json
from pathlib import Path
from typing import Union
from datetime import datetime as dt
import shutil

import numpy as np
import pandas as pd
import pynwb
import utils
from hdmf_zarr import NWBZarrIO
from pynwb import NWBHDF5IO
from aind_nwb_utils import utils as nwb_utils
from aind_data_schema.core.processing import DataProcess
from aind_data_schema.base import AindGeneric
from aind_data_schema_models.process_names import ProcessName


DEFAULT_RUNNING_SPEED_UNITS = {
    "velocity": "cm/s",
    "vin": "V",
    "vsig": "V",
    "rotation": "radians",
}


data_folder = Path("../data/")
scratch_folder = Path("../scratch/")
results_folder = Path("../results/")


def write_data_process(
    metadata: dict,
    h5_path: Union[str, Path],
    nwb_path: Union[str, Path],
    output_dir: Union[str, Path],
    start_time: dt,
    end_time: dt,
) -> None:
    """Writes output metadata to plane processing.json

    Parameters
    ----------
    metadata: dict
        parameters from suite2p motion correction
    h5_path: str
        path to h5
    nwb_path: str
        path to the nwb
    """
    if isinstance(h5_path, Path):
        h5_path = str(h5_path)
    if isinstance(nwb_path, Path):
        nwb_path = str(nwb_path)
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    data_proc = DataProcess(
        name=ProcessName.OTHER,
        software_version=os.getenv("VERSION", ""),
        start_date_time=start_time.isoformat(),
        end_date_time=end_time.isoformat(),
        parameters=AindGeneric(**metadata),
        input_location=str(h5_path),
        output_location=str(nwb_path),
        code_url="https://github.com/AllenNeuralDynamics/"
        "NWB-Packaging-Running-Capsule/code/",
        code_version=os.getenv("VERSION", ""),
        notes="Bci behavior stimulus table",
    )
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    with open(
        output_dir / "running-nwb-packaging_data_process.json", "w"
    ) as f:
        json.dump(json.loads(data_proc.model_dump_json()), f, indent=4)


def extract_running_speeds(
    frame_times: np.array,
    dx_deg: np.array,
    wheel_radius: float,
    subject_position: float,
    use_median_duration: bool = False,
) -> pd.DataFrame:
    """Extract running speeds from raw running wheel data

    Parameters
    ----------
    frame_times : np.array
        frame times from the sync dataset
    dx_deg : np.array
        change in orientation of the running wheel in degrees
    wheel_radius : float
        radius of the running wheel
    subject_position : float
        position of the subject on the running wheel
    use_median_duration : bool, optional
        normalize velocity to median, by default False

    Returns
    -------
    pd.DataFrame
        DataFrame containing the start and end times of each interval, the
        velocity of the running wheel, and the net rotation of the wheel
    """
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
    linear_velocity = utils.angular_to_linear_velocity(
        angular_velocity, radius
    )

    print("lengths of start times:",len(start_times),"end times",len(end_times))
    print("velocity:",len(linear_velocity),"rotation:",len(dx_rad))

    # there might be one too few recorded times, for running its fine to just truncate 
    if len(start_times) == len(linear_velocity)-1:
        linear_velocity = linear_velocity[:-1]
        print(f"one extra velocity time. Truncating to {len(linear_velocity)}")
    if len(start_times) == len(dx_rad)-1:
        dx_rad = dx_rad[:-1]
        print(f"one extra rotation time. Truncating to {len(dx_rad)}")
    df = pd.DataFrame(
        {
            "start_time": start_times,
            "end_time": end_times,
            "velocity": linear_velocity,
            "net_rotation": dx_rad,
        }
    )
    sudden_drop = (
        (
            abs(df["velocity"].shift(1)) > 0.5
        )  # Previous velocity is significantly non-zero
        & np.isclose(
            df["velocity"], 0.0, atol=1e-3
        )  # Current velocity is near zero
        & (
            abs(df["velocity"].shift(-1)) > 0.5
        )  # Following velocity is significantly non-zero
    )

    # Remove rows with near-zero velocity only if
    # they are flanked by significant non-zero values
    df = df[~sudden_drop]

    return df


def add_running_speed_to_nwbfile(
    nwbfile: Union[NWBZarrIO, NWBHDF5IO],
    running_speed: np.array,
    units: dict = None,
):
    """Add running speed data to an NWB file

    Parameters
    ----------
    nwbfile : NWBFile
        NWB file to add running speed data to
    running_speed : pd.DataFrame
        DataFrame containing running speed data
    units : dict, optional
        units for the running speed data, by default None

    Returns
    -------
    NWBFile
        NWB file with running speed data added
    """
    if units is None:
        units = DEFAULT_RUNNING_SPEED_UNITS

    running_mod = pynwb.ProcessingModule("running", "running speed data")
    nwbfile.add_processing_module(running_mod)

    running_speed_timeseries = pynwb.base.TimeSeries(
        name="running_speed",
        timestamps=running_speed["start_time"].values,
        data=running_speed["velocity"].values,
        unit=units["velocity"],
    )

    rotation_timeseries = pynwb.base.TimeSeries(
        name="running_wheel_rotation",
        timestamps=running_speed_timeseries,
        data=running_speed["net_rotation"].values,
        unit=units["rotation"],
    )

    running_mod.add_data_interface(running_speed_timeseries)
    running_mod.add_data_interface(rotation_timeseries)

    return nwbfile


def add_raw_running_data_to_nwbfile(
    nwbfile: Union[NWBHDF5IO, NWBZarrIO],
    raw_running_data: dict,
    units: dict = None,
):
    """Add raw running data to an NWB file

    Parameters
    ----------
    nwbfile : NWBFile
        NWB file to add running speed data to
    raw_running_data : dict
        dictionary containing raw running data
    units : dict, optional

    Returns
    -------
    NWBFile
        NWB file with raw running data added
    """
    if units is None:
        units = DEFAULT_RUNNING_SPEED_UNITS

    raw_rotation_timeseries = pynwb.base.TimeSeries(
        name="raw_running_wheel_rotation",
        timestamps=np.array(raw_running_data["frame_time"]),
        data=raw_running_data["dx"].values,
        unit=units["rotation"],
    )

    vsig_ts = pynwb.base.TimeSeries(
        name="running_wheel_signal_voltage",
        timestamps=raw_rotation_timeseries,
        data=raw_running_data["vsig"].values,
        unit=units["vsig"],
    )

    vin_ts = pynwb.base.TimeSeries(
        name="running_wheel_supply_voltage",
        timestamps=raw_rotation_timeseries,
        data=raw_running_data["vin"].values,
        unit=units["vin"],
    )

    nwbfile.add_acquisition(raw_rotation_timeseries)
    nwbfile.add_acquisition(vsig_ts)
    nwbfile.add_acquisition(vin_ts)

    return nwbfile


def get_running_data(
    stim_file: Union[Path, str], sync_dataset: pd.DataFrame
) -> pd.DataFrame:
    """Get running data from a stimulus file and sync dataset

    Parameters
    ----------
    stim_file : Union[Path, str]
        path to the stimulus file
    sync_dataset : pd.DataFrame
        sync dataset

    Returns
    -------
    pd.DataFrame
        DataFrame containing running speeds
    """
    # Why the rising edge? See Sweepstim.update in camstim. This method does:
    # 1. updates the stimuli
    # 2. updates the "items", causing a running speed sample to be acquired
    # 3. sets the vsync line high
    # 4. flips the buffer
    frame_times = utils.get_edges(
        sync_dataset,
        "rising",
        ("frames", "stim_vsync", "vsync_stim"),
        units="seconds",
    )

    num_raw_timestamps = len(frame_times)
    logging.info(num_raw_timestamps)
    trimmed_times = utils.trim_discontiguous_times(frame_times)
    logging.info(len(trimmed_times))

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
        wheel_radius=8.255,
        subject_position=2 / 3,
        use_median_duration=True,
    )

    raw_data = pd.DataFrame(
        {"vsig": vsig, "vin": vin, "frame_time": frame_times, "dx": dx_deg}
    )
    return velocities, raw_data


def parse_args():
    """Get command line arguments

    Returns:
        argparse.Namespace: command line arguments
    """

    parser = argparse.ArgumentParser(
        description="Package running speed data into an NWB file"
    )
    parser.add_argument(
        "--use_input_nwb",
        type=str,
        help="Whether or to use the NWB at --input_nwb_path or to create a new one from --input_behavior_dir",
        default = 'False'
    )
    parser.add_argument(
        "--input_nwb_dir",
        type=str,
        help="Path within ../data to the folder containing the nwb file",
        default="nwb",
    )
    parser.add_argument(
        "--input_behavior_dir",
        type=str,
        help="Path to the folder containing the pkl and sync files",
        default="session/behavior",
    )
    return parser.parse_args()


def run():
    """basic run function"""
    start_time = dt.now()
    args = parse_args()
    input_behavior_dir = data_folder / args.input_behavior_dir
    use_input_nwb = args.use_input_nwb
    input_nwb_dir = data_folder / args.input_nwb_dir

    if args.use_input_nwb in ('t','T','true','True'):
        print('INPUT NWB DIR', input_nwb_dir)
        assert input_nwb_dir.exists(), "Input NWB Dir does not exist"
        nwb_path = next(input_nwb_dir.rglob("*.nwb"))
        # determine if file is zarr or hdf5, and copy it to results

        result_nwb_path = results_folder / nwb_path.name
        if nwb_path.is_dir():
            assert (
                nwb_path / ".zattrs"
            ).is_file(), f"{nwb_path.name} is not a valid Zarr folder"
            io_class = NWBZarrIO
            shutil.copytree(nwb_path, result_nwb_path, dirs_exist_ok=True)
        else:
            io_class = NWBHDF5IO
            shutil.copyfile(nwb_path, result_nwb_path)
        nwb_path = result_nwb_path
    else:
        io_class = NWBZarrIO
        nwb_file_obj = nwb_utils.create_base_nwb_file(input_behavior_dir.parent)
        nwb_name = nwb_file_obj.session_id
        nwb_path = f"/results/{nwb_name}.nwb"
        with io_class(str(nwb_path), "w") as io:
            io.write(nwb_file_obj)
    print("Using NWB:", nwb_path)

    print("INPUT BEHAVIOR DIR", input_behavior_dir)
    assert input_behavior_dir.exists(), "Input  Dir does not exist"
    sync_paths = list(input_behavior_dir.rglob("*.h5"))
    if len(sync_paths) == 0:
        sync_paths = list(input_behavior_dir.rglob("*.sync"))
    sync_path = sync_paths[0]
    print("Using sync:", sync_path)

    stim_pkl_files = [
        p for p in input_behavior_dir.iterdir() if p.name.endswith(".stim.pkl")
    ]
    behavior_pkl_files = [
        p
        for p in input_behavior_dir.iterdir()
        if p.name.endswith(".behavior.pkl")
    ]
    if len(stim_pkl_files) == 0 and len(behavior_pkl_files) == 0:
        stim_pkl_files = [
            p for p in input_behavior_dir.iterdir() if p.name.endswith(".pkl")
        ]
    if len(stim_pkl_files) != 1:
        if len(stim_pkl_files) == 2:
            pkl_path = stim_pkl_files[0]
        elif len(behavior_pkl_files) != 1:
            raise Exception(
                "Expected exactly one pkl file match."
                f"Found\n stim_pkl files: {stim_pkl_files}\n"
                f" behavior pkl files: {behavior_pkl_files}"
            )
        else:
            pkl_path = behavior_pkl_files[0]
    else:
        pkl_path = stim_pkl_files[0]
    print("Using pkl:", pkl_path)

    logging.info(
        f"pkl file: {pkl_path},\nsync file: {sync_path},\nnwb file: {nwb_path}"
    )
    stim_file = pd.read_pickle(str(pkl_path))
    sync_dataset = utils.load_sync(str(sync_path))

    velocities, raw_data = get_running_data(stim_file, sync_dataset)
    io = io_class(str(nwb_path), "r+")
    nwb_file = io.read()
    nwb_file = add_running_speed_to_nwbfile(nwb_file, velocities)
    nwb_file = add_raw_running_data_to_nwbfile(nwb_file, raw_data)
    io.write(nwb_file)
    io.close()
    end_time = dt.now()
    write_data_process(
        h5_path=sync_path,
        nwb_path=nwb_path,
        output_dir=results_folder,
        start_time=start_time,
        end_time=end_time.now(),
        metadata={
            "wheel_radius": 8.255,
            "subject_position": 2 / 3,
            "use_median_duration": True,
        },
    )
    logging.info("Running speed packaging completed successfully.")


if __name__ == "__main__":
    run()
