from pathlib import Path
import pandas as pd
from nptdms import TdmsFile
import datetime

if __name__ == "__main__":

    ###  READ TDMS FILES & write to hdf 5
    ###  ----------------

    # tdmsfile_path = Path(r"D:\#1 - 500h CC wo UI")
    # tdmsfile_path = Path(r"D:\#2 - 500h CC w UI 4h")
    tdmsfile_path = Path(r"D:\#3 - 500h CC wo UI")
    #
    #
    # channelgroups_path = Path(r"channel_groups_20240214 (1).json")

    for p in tdmsfile_path.glob("*.tdms"):
        full_data = []
        print(p)

        # https://nptdms.readthedocs.io/en/stable/quickstart.html
        print("start:", datetime.datetime.now())
        tdms_file = TdmsFile.read(p)
        print("finish read:", datetime.datetime.now())

        for group in tdms_file.groups():
            group_name = group.name
            if group_name == "data":
                print("group name: ", group_name)
                for channel in group.channels():
                    if channel.name in [
                        "Time",
                        "StackV (V)",
                        "StackV1 (V)",
                        "StackV2 (V)",
                        "StackV3 (V)",
                        "StackV4 (V)",
                        "StackV5 (V)",
                        "SUPPLY.Current (A)",
                    ]:
                        channel_name = channel.name
                        print("channel name: ", channel_name)
                        # Access dictionary of properties:
                        properties = channel.properties
                        # Access numpy array of data for channel:
                        data = channel[:]
                        # Access a subset of data
                        # data_subset = channel[100:200]
                        dataset = pd.DataFrame(columns=[channel_name], data=data)
                        full_data.append(dataset)

            print("finish loop:", datetime.datetime.now())
        df = pd.concat(full_data, axis=1)
        df.to_hdf(str(p) + ".hdf", key="df")
        # df.to_pickle(str(p)+".pkl")

    data_full = []
    for file in tdmsfile_path.glob("*.hdf"):
        print("read ", file)
        # f = h5py.File(file, 'r')
        df = pd.read_hdf(file, key="df")
        data_full.append(df)
    df = pd.concat(data_full)
