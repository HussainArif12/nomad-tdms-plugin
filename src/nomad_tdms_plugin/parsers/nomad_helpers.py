import math
import os

import h5py
import pandas as pd
from nomad.datamodel.context import ClientContext
from nptdms import ChannelObject, GroupObject, RootObject, TdmsFile, TdmsWriter


def nan_equal(a, b):
    """
    Compare two values with NaN values.
    """
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    elif isinstance(a, dict) and isinstance(b, dict):
        return dict_nan_equal(a, b)
    elif isinstance(a, list) and isinstance(b, list):
        return list_nan_equal(a, b)
    else:
        return a == b


def list_nan_equal(list1, list2):
    """
    Compare two lists with NaN values.
    """
    if len(list1) != len(list2):
        return False
    for a, b in zip(list1, list2):
        if not nan_equal(a, b):
            return False
    return True


def dict_nan_equal(dict1, dict2):
    """
    Compare two dictionaries with NaN values.
    """
    if set(dict1.keys()) != set(dict2.keys()):
        return False
    for key in dict1:
        if not nan_equal(dict1[key], dict2[key]):
            return False
    return True


def get_reference(upload_id, entry_id):
    return f"../uploads/{upload_id}/archive/{entry_id}"


def get_entry_id(upload_id, filename):
    from nomad.utils import hash

    return hash(upload_id, filename)


def get_hash_ref(upload_id, filename):
    return f"{get_reference(upload_id, get_entry_id(upload_id, filename))}#data"


def create_archive(
    zyklus_nr,
    typ,
    temp,
    zustand,
    datum_str,
    cycle_data,
    entry_dict,
    context,
    archive,
    filename,
    file_type,
    logger,
    *,
    overwrite: bool = False,
):
    file_exists = context.raw_path_exists(filename)

    dicts_are_equal = None
    if isinstance(context, ClientContext):
        print("Context for create archive is ClientContext")
        return None

    if not file_exists or overwrite or dicts_are_equal:
        with archive.m_context.raw_file(filename, "wb") as file:
            with TdmsWriter(file.name) as tdms_writer:
                root = RootObject(
                    properties={
                        "Zyklus": int(zyklus_nr),
                        "Typ": str(typ),
                        "Temp": str(temp),
                        "Zustand": str(zustand),
                        "Datum": datum_str,
                    }
                )
                all_objects = [root]
                min_time, max_time, total_points = None, None, 0
                for gname, group in cycle_data.items():
                    gobj = GroupObject(gname)  # properties=None
                    all_objects.append(gobj)

                    for cname, df in group.items():
                        if df.empty:
                            continue

                        # globale Dauer und Punkte zählen
                        tmin, tmax = df["time"].min(), df["time"].max()
                        if min_time is None or tmin < min_time:
                            min_time = tmin
                        if max_time is None or tmax > max_time:
                            max_time = tmax
                        total_points += len(df)

                        values = df["value"].values
                        ts = df["time"].astype("datetime64[ns]").values
                        ch_val = ChannelObject(gname, f"{cname}.Current Value", values)
                        ch_ts = ChannelObject(gname, f"{cname}.Timestamp", ts)
                        all_objects.extend([ch_val, ch_ts])

                tdms_writer.write_segment(all_objects)

        context.upload.process_updated_raw_file(filename, allow_modify=True)

    elif file_exists and not overwrite and not dicts_are_equal:
        logger.error(
            f"{filename} archive file already exists. "
            f"You are trying to overwrite it with a different content. "
            f"To do so, remove the existing archive and click reprocess again."
        )
    return get_hash_ref(context.upload_id, filename)


def convert_to_hdf(
    context,
    archive,
    file_path,
    logger,
    overwrite=False,
):

    print("here")
    filename = os.path.basename(file_path[-1]).split("/")[-1]
    filename = filename.rsplit(".", 1)[0] + ".hdf"
    filename = "output_file.hdf"
    file_exists = context.raw_path_exists(filename)

    full_data = []
    if not file_exists or overwrite:
        tdms_file = TdmsFile.read(file_path)
        for group in tdms_file.groups():
            group_name = group.name

            for channel in group.channels():
                channel_name = channel.name
                # Access dictionary of properties:
                properties = channel.properties
                # Access numpy array of data for channel:
                data = channel[:]
                # Access a subset of data
                # data_subset = channel[100:200]
                dataset = pd.DataFrame(
                    columns=[f"{group_name}/{channel_name}"], data=data
                )
                full_data.append(dataset)

        with archive.m_context.raw_file(filename, "wb") as file:
            if full_data:
                num_array_length = len(full_data)
                df = pd.concat(full_data, axis=1)
                df.dropna(inplace=True)
                print("length of dataframe from df -> tdms:", len(df))
                with h5py.File(file.name) as hdf:
                    for column in df.columns:
                        values = df[column]
                        group = hdf.create_group(column)
                        group.create_dataset("value", data=values)
                        group.create_dataset("time", data=num_array_length)
                        group.attrs["axes"] = "time"
                        group.attrs["signal"] = "value"
                        group.attrs["NX_class"] = "NXdata"

        # archive.m_context.upload.process_updated_raw_file(filename, allow_modify=True)
    elif file_exists and not overwrite:
        logger.error(
            f"{filename} archive file already exists. "
            f"You are trying to overwrite it with a different content. "
            f"To do so, remove the existing archive and click reprocess again."
        )
    return get_hash_ref(context.upload_id, filename)


def convert_another_hdf(archive, mainfile):
    filename = os.path.basename(mainfile[0]).split("/")[-1]
    filename = filename.rsplit(".", 1)[0] + ".hdf"
    full_data = []
    tdms_file = TdmsFile.read(mainfile[0])
    for group in tdms_file.groups():
        group_name = group.name
        for channel in group.channels():
            channel_name = channel.name
            data = channel[:]
            dataset = pd.DataFrame(columns=[f"{group_name}/{channel_name}"], data=data)
            full_data.append(dataset)

    df = pd.concat(full_data, axis=1)
    print(df)
    num_array_length = len(df)
    df.dropna(inplace=True)
    with archive.m_context.raw_file(filename, "w") as newfile:
        with h5py.File(newfile.name, "w") as hdf:
            for column in df.columns:
                values = df[column]
                if pd.api.types.is_datetime64_any_dtype(values):
                    # Option A: Save as strings (best for readability)
                    values = values.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").values.astype(
                        "S"
                    )
                    print(values)
                group = hdf.create_group(column)
                try:
                    group.create_dataset("value", data=values)
                    group.create_dataset("time", data=num_array_length)
                except Exception:
                    print(values)
