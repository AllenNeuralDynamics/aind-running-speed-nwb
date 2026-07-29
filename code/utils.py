from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import re
import warnings

DEGREES_TO_RADIANS = np.pi / 180.0


def check_encoder(parent, key):
    if len(parent["encoders"]) != 1:
        return False
    if key not in parent["encoders"][0]:
        return False
    if len(parent["encoders"][0][key]) == 0:
        return False
    return True



def running_from_stim_file(stim_file, key, expected_length):
    if 'bonsai' in stim_file:
        radius = stim_file['config']['DigitalEncoder']['radius_cm']
        encoder_dict = _reconstruct_encoder_from_logger(stim_file['bonsai']['logger'], radius=radius)
        return encoder_dict[key]

    if "behavior" in stim_file["items"] and check_encoder(
        stim_file["items"]["behavior"], key
    ):
        return stim_file["items"]["behavior"]["encoders"][0][key][:]
    if "foraging" in stim_file["items"] and check_encoder(
        stim_file["items"]["foraging"], key
    ):
        return stim_file["items"]["foraging"]["encoders"][0][key][:]
    if key in stim_file:
        print("key in stim file is:", key)
        return stim_file[key][:]

    warnings.warn(f"unable to read {key} from this stimulus file")
    return np.ones(expected_length) * np.nan


def degrees_to_radians(degrees):
    return np.array(degrees) * DEGREES_TO_RADIANS


def angular_to_linear_velocity(angular_velocity, radius):
    return np.multiply(angular_velocity, radius)


def load_sync(path):
    """
    Loads an hdf5 sync dataset.

    Parameters
    ----------
    path : str
        Path to hdf5 file.

    """
    dfile = h5py.File(path, "r")
    return dfile


def get_edges(
    sync_file: h5py.File,
    kind: str,
    keys: Union[str, Sequence[str]],
    units: str = "seconds",
    permissive: bool = False,
) -> Optional[np.ndarray]:
    """Utility function for extracting edge times from a line

    Parameters
    ----------
    kind : One of "rising", "falling", or "all". Should this method return
        timestamps for rising, falling or both edges on the appropriate
        line
    keys : These will be checked in sequence. Timestamps will be returned
        for the first which is present in the line labels
    units : one of "seconds", "samples", or "indices". The returned
        "time"stamps will be given in these units.
    raise_missing : If True and no matching line is found, a KeyError will
        be raised

    Returns
    -------
    An array of edge times. If raise_missing is False and none of the keys
        were found, returns None.

    Raises
    ------
    KeyError : none of the provided keys were found among this dataset's
        line labels

    """

    if isinstance(keys, str):
        keys = [keys]

    print(keys)

    for line in keys:
        try:
            if kind == "falling":
                return get_falling_edges(sync_file, line, units)
            elif kind == "rising":
                return get_rising_edges(sync_file, line, units)
            elif kind == "all":
                return np.sort(
                    np.concatenate(
                        [
                            get_edges(sync_file, "rising", keys, units),
                            get_edges(sync_file, "falling", keys, units),
                        ]
                    )
                )
        except ValueError:
            continue

    if not permissive:
        raise KeyError(f"none of {keys} were found in this dataset's line labels")


def get_rising_edges(sync_file, line, units="samples"):
    """
    Returns the counter values for the rizing edges for a specific bit or
        line.

    Parameters
    ----------
    line : str
        Line for which to return edges.

    """
    meta_data = get_meta_data(sync_file)
    bit = line_to_bit(sync_file, line)
    changes = get_bit_changes(sync_file, bit)
    return get_all_times(sync_file, meta_data, units)[np.where(changes == 1)]


def trim_discontiguous_times(times: np.ndarray, threshold=100) -> np.ndarray:
    """
    If the time sequence is discontigous,
    detect the first instance occurance and trim off the tail of the sequence

    Parameters
    ----------
    times : frame times

    Returns
    -------
    trimmed frame times
    """

    times = np.array(times)
    intervals = np.diff(times)

    med_interval = np.median(intervals)
    interval_threshold = med_interval * threshold

    gap_indices = np.where(intervals > interval_threshold)[0]

    # A special case for when the first element is a discontiguity
    if np.abs(intervals[0]) > interval_threshold:
        gap_indices = [0]

    if len(gap_indices) == 0:
        return times

    return times[: gap_indices[0] + 1]


def get_synchronized_frame_times(
    session_sync_file: Path,
    sync_line_label_keys: Tuple[str, ...],
    drop_frames: Optional[List[int]] = None,
    trim_after_spike: bool = True,
) -> pd.Series:
    """Get experimental frame times from an experiment session sync file.

    1. Get rising edges from the sync dataset
    2. Occasionally an extra set of frame times are acquired after the rest of
        the signals. These are manifested by a discontiguous time sequence.
        We detect and remove these.
    3. Remove dropped frames

    Parameters
    ----------
    session_sync_file : Path
        Path to an ephys session sync file.
        The sync file contains rising/falling edges from a daq system which
        indicates when certain events occur (so they can be related to
        each other).
    sync_line_label_keys : Tuple[str, ...]
        Line label keys to get times for. See class attributes of
        allensdk.brain_observatory.sync_dataset.Dataset for a listing of
        possible keys.
    drop_frames : List
        frame indices to be removed from frame times
    trim_after_spike : bool = True
        If True, will call trim_discontiguous_times on the frame times
        before returning them, which will detect any spikes in the data
        and remove all elements for the list which come after the spike.

    Returns
    -------
    pd.Series
        An array of times when eye tracking frames were acquired.
    """
    times = get_edges(session_sync_file, "rising", sync_line_label_keys, units="seconds")

    times = trim_discontiguous_times(times) if trim_after_spike else times
    if drop_frames is not None:
        times = [t for ix, t in enumerate(times) if ix not in drop_frames]

    return pd.Series(times)


def get_meta_data(sync_file):
    """
    Returns the metadata for the sync file.

    """
    meta_data = eval(sync_file["meta"][()])
    return meta_data


def line_to_bit(sync_file, line):
    """
    Returns the bit for a specified line.  Either line name and number is
        accepted.

    Parameters
    ----------
    line : str
        Line name for which to return corresponding bit.

    """
    line_labels = get_line_labels(sync_file)

    if type(line) is int:
        return line
    elif type(line) is str:
        return line_labels.index(line)
    else:
        raise TypeError("Incorrect line type.  Try a str or int.")


def get_line_labels(sync_file):
    """
    Returns the line labels for the sync file.

    """
    meta_data = get_meta_data(sync_file)
    line_labels = meta_data["line_labels"]
    return line_labels


def get_bit_changes(sync_file, bit):
    """
    Returns the first derivative of a specific bit.
        Data points are 1 on rising edges and 255 on falling edges.

    Parameters
    ----------
    bit : int
        Bit for which to return changes.

    """
    bit_array = get_sync_file_bit(sync_file, bit).astype(np.int64)
    return np.ediff1d(bit_array, to_begin=0)


def get_sync_file_bit(sync_file, bit):
    return get_bit(get_all_bits(sync_file), bit)


def get_bit(uint_array, bit):
    """
    Returns a bool array for a specific bit in a uint ndarray.

    Parameters
    ----------
    uint_array : (numpy.ndarray)
        The array to extract bits from.
    bit : (int)
        The bit to extract.

    """
    return np.bitwise_and(uint_array, 2**bit).astype(bool).astype(np.uint8)


def get_all_bits(sync_file):
    """
    Returns the data for all bits.

    """
    return sync_file["data"][()][:, -1]


def get_all_times(sync_file, meta_data, units="samples"):
    """
    Returns all counter values.

    Parameters
    ----------
    units : str
        Return times in 'samples' or 'seconds'

    """
    if meta_data["ni_daq"]["counter_bits"] == 32:
        times = sync_file["data"][()][:, 0]
    else:
        times = times
    units = units.lower()
    if units == "samples":
        return times
    elif units in ["seconds", "sec", "secs"]:
        freq = get_sample_freq(meta_data)
        return times / freq
    else:
        raise ValueError("Only 'samples' or 'seconds' are valid units.")


def get_sample_freq(meta_data):
    try:
        return float(meta_data["ni_daq"]["sample_freq"])
    except KeyError:
        return float(meta_data["ni_daq"]["counter_output_freq"])


def _reconstruct_encoder_from_logger(logger_rows, total_frames=None, radius=None):
    """Reconstruct CAMSTIM-style encoder data from Bonsai logger rows.

    Wheel event format example Value field:
        'Wheel-Index-42793322-Count-4785-Deg-210.2783203125'

    We parse per-frame events, derive per-frame dtheta (dx) using logic similar
    to DigitalBehaviorEncoder, propagate last values for frames without events,
    and compute derived vsig and optional distance.
    """
    wheel_pattern = re.compile(
        r'^Wheel-Index-(\d+)-Count-(-?\d+)-Deg-([-\d\.eE]+)$'
    )   
    start_time_pattern = re.compile(r'^START$')
    end_time_pattern = re.compile(r'^END$')

    events_by_frame = {}
    max_frame = 0
    global_ref_timestamps = 0 

    for row in logger_rows:
        try:
            frame = int(row.get('Frame', '0'))
            if frame > max_frame:
                max_frame = frame
            val = row.get('Value', '')
            m = wheel_pattern.match(val)
            if m:
                idx = int(m.group(1))
                count = int(m.group(2))
                deg = float(m.group(3))
                events_by_frame[frame] = {'index': idx, 'count': count, 'deg': deg}
            elif start_time_pattern.match(val):
                # We save the start frame to align timestamps later 
                global_ref_timestamps = float(row.get('Timestamp'))
                print("Found START marker at frame %d for encoder reconstruction, timestamp %s" % (frame, global_ref_timestamps))
            elif end_time_pattern.match(val):
                print("Found END marker at frame %d for encoder reconstruction" % frame) 
                end_session = float(row.get('Timestamp'))

        except Exception:
            print("Error parsing logger row for encoder reconstruction: %s" % row)
            continue

    # Determine number of frames to allocate
    n_frames = int(max_frame) + 1
    if total_frames and total_frames > n_frames:
        n_frames = int(total_frames)

    dx = np.zeros(n_frames, dtype=np.float32)
    degrees = np.zeros(n_frames, dtype=np.float64)
    counts = np.zeros(n_frames, dtype=np.float32)
    timestamps = np.zeros(n_frames, dtype=np.float32)

    last_deg = None
    last_index = None
    last_dtheta = 0.0

    # For timestamps, build map first
    frame_time = {}
    for row in logger_rows:
        try:
            f = int(row.get('Frame', '0'))
            if f not in frame_time:
                frame_time[f] = float(row.get('Timestamp', '0'))
        except Exception:
            pass

    in_evt_count = 0
    not_evt_count = 0

    print("n frames!",n_frames)
    for f in range(n_frames):
        evt = events_by_frame.get(f)
        if evt:
            in_evt_count += 1
            idx = evt['index']
            deg = evt['deg']
            if last_deg is None or last_index is None:
                dtheta = 0.0
            else:
                if idx == last_index:
                    dtheta = last_dtheta  # reuse
                else:
                    try:
                        dtheta = (deg - last_deg) / (idx - last_index)
                    except ZeroDivisionError:
                        dtheta = 0.0
            last_dtheta = dtheta
            last_deg = deg
            last_index = idx
            degrees[f] = deg
            dx[f] = dtheta
            counts[f] = evt['count']
        else:
            not_evt_count += 1
            # propagate degree by integrating last_dtheta
            if f > 0:
                degrees[f] = degrees[f-1] + last_dtheta
                counts[f] = counts[f-1]
            else:
                degrees[f] = 0.0
                counts[f] = 0.0
            dx[f] = last_dtheta
        # previous frame timestamp or we substract the last known deltatime
        timestamps[f] = frame_time.get(f, timestamps[f-1] if f > 0 else timestamps[f]-0.016)

    # If some degrees slots remain zero but we had last_deg set, retroactively fill preceding frames
    # (Not strictly necessary; we already integrate above.)

    # Compute vsig & vin
    vsig = (degrees % 360.0) * (5.0 / 360.0)
    vin = np.full(n_frames, 5.0, dtype=np.float32)

    # Distance if radius available
    # radius = self.config.get('digital_encoder', {}).get('radius_cm') \
    #     or self.config.get('encoder', {}).get('radius_cm')
    if radius is not None:
        try:
            radius = float(radius)
            distance = dx * (np.pi/180.0) * radius
        except Exception:
            distance = None
    else:
        distance = None

    # We restrict the arrays between global_ref_timestamps and end_session if available
    if global_ref_timestamps and end_session:
        timestamps_float = timestamps.astype(np.float64, copy=False)
        kept_frames = ((timestamps_float >= float(global_ref_timestamps)) &
                        (timestamps_float <= float(end_session)))
        n_kept = np.sum(kept_frames)
        print("Trimming encoder arrays from %d to %d frames based on START/END timestamps" % (n_frames, n_kept))
        dx = dx[kept_frames]
        degrees = degrees[kept_frames]
        vin = vin[kept_frames]
        vsig = vsig[kept_frames]
        counts = counts[kept_frames]
        timestamps = timestamps[kept_frames]
        distance = distance[kept_frames] if distance is not None else None

    encoder_dict = {
        'dx': dx.astype(np.float32),
        'gain': 1.0,
        'items': {},
        'unpickleable': [],
        'value': float(degrees[-1]) if len(degrees) else 0.0,
        'vin': vin,
        'vsig': vsig.astype(np.float32),
        'analog_encoder': False,
        'counts': counts.astype(np.float32),
        'timestamp': timestamps.astype(np.float32)-global_ref_timestamps
    }
    if distance is not None:
        encoder_dict['distance'] = distance.astype(np.float32)

    return encoder_dict
