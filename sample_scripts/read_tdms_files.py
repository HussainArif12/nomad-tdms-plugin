from pathlib import Path
import pandas as pd
from nptdms import TdmsFile
import datetime
from nptdms import TdmsWriter, ChannelObject
import os

if __name__ == "__main__":

    ###  READ TDMS FILES & write to hdf 5
    ###  ----------------

    # tdmsfile_path = Path(r"D:\#1 - 500h CC wo UI")
    # tdmsfile_path = Path(r"D:\#2 - 500h CC w UI 4h")
    tdmsfile_path = Path(
        r"/home/student/projects/nomad-tdms-plugin/tests/example_uploads/PROCESS_DATA_STORAGE_2025-09-03_14-01-21.tdms"
    )
    #
    #
    # channelgroups_path = Path(r"channel_groups_20240214 (1).json")
    filename = os.path.basename(tdmsfile_path).split("/")[-1]

    filename = filename.rsplit(".", 1)[0] + ".hdf"
    full_data = []

    # https://nptdms.readthedocs.io/en/stable/quickstart.html
    print("start:", datetime.datetime.now())
    tdms_file = TdmsFile.read(tdmsfile_path)
    # group = tdms_file["PROCESSIMAGE"]
    # channel = group["channel name"]
    # channel_data = channel[:]
    # channel_properties = channel.properties
    print("finish read:", datetime.datetime.now())
    # summary counts
    groups = list(tdms_file.groups())
    num_groups = len(groups)
    num_channels = sum(1 for g in groups for _ in g.channels())
    channels_per_group = {g.name: sum(1 for _ in g.channels()) for g in groups}
    total_samples = sum(len(ch[:]) for g in groups for ch in g.channels())

    print(f"TDMS summary: {num_groups} groups, {num_channels} channels")
    for name, cnt in channels_per_group.items():
        print(f"  Group '{name}': {cnt} channels")
    print(f"Total data samples across all channels: {total_samples}")

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
            dataset = pd.DataFrame(columns=[f"{group_name}/{channel_name}"], data=data)
            full_data.append(dataset)

    print("length of full_data ", len(full_data))
    if len(full_data):
        df = pd.concat(full_data, axis=1)
        print(df)
        df.dropna(inplace=True)
        print("length of dataframe from df -> tdms:", len(df))
        df.to_hdf(filename, key="df")
        df.to_pickle(str(tdmsfile_path) + ".pkl")

    print("Now reading back hdf files")
    data_full = []
    hd_file = "/home/student/projects/nomad-tdms-plugin/tests/example_uploads/PROCESS_DATA_STORAGE_2025-09-03_14-01-21.tdms.hdf"
    df = pd.read_hdf(hd_file, key="df")

    # print("Now writing.. ")
    # with TdmsWriter("./tdms_file.tdms") as tdms_writer:
    #     for column_name in df.columns:
    #         group_name, channel_name = column_name.split("/")
    #         data = df[column_name].values
    #         channel = ChannelObject(group_name, channel_name, data)
    #         tdms_writer.write_segment([channel])
    # for file in tdmsfile_path.glob("*.hdf"):
    #     print("read ", file)
    #     # f = h5py.File(file, 'r')
    #     df = pd.read_hdf(file, key="df")
    #     data_full.append(df)
    # df = pd.concat(data_full)
