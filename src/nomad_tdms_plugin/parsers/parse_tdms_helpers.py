import os
import pandas as pd
from nptdms import TdmsFile, TdmsWriter, ChannelObject, RootObject, GroupObject
import time
from .nomad_helpers import create_archive

OUTPUT_FOLDER = r"./outputs"
PRUEFLINGSNAME = "636"  # manuell setzen
BASENAME = f"NVM5evo-{PRUEFLINGSNAME}-11"

# Ausnahmen bleiben voll erhalten
EXCEPTION_INDICES = set([501, 502, 503] + list(range(700, 718)) + [800, 900])

INDEX_MAPPING = {
    620: ("B", "-30°C", "Referenzmessung"),
    623: ("B", "-30°C", "Froststart"),
    622: ("B", "-20°C", "Froststart"),
    602: ("D", "-30°C", "Referenzmessung"),
    603: ("D", "-30°C", "Froststart"),
    680: ("H", "-30°C", "Referenzmessung"),
    683: ("H", "-30°C", "Froststart"),
}

REQUIRED_INDICES = {
    99,
    100,
    102,
    103,
    104,
    201,
    202,
    211,
    212,
    213,
    214,
    215,
    216,
    217,
    218,
    219,
    220,
    221,
    222,
    223,
    224,
    225,
    226,
    227,
    228,
    229,
    230,
    231,
    232,
    240,
    241,
    250,
    251,
    252,
    253,
    254,
    255,
    256,
    257,
    258,
    259,
    260,
    299,
    300,
    501,
    502,
    503,
    600,
    601,
    700,
    701,
    702,
    703,
    704,
    705,
    706,
    709,
    710,
    800,
    900,
}


# ========== Utilities ==========


def timed(label, fn, logger, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    logger.info("NewSchema.parse", parameter=f"[{label}] {dt:0.2f}s")
    return out


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def load_tdms_file(file_path):
    """Load a single TDMS file and return it as a list for compatibility with rest of code."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TDMS file not found: {file_path}")
    if not file_path.endswith(".tdms"):
        raise ValueError(f"File must be a .tdms file: {file_path}")
    return [file_path]


def get_file_timerange(path):
    try:
        tdms = TdmsFile.read(path)
        group = tdms["PROCESSIMAGE"]
    except Exception as e:
        # Im Zweifel Datei nicht ausschließen
        print(
            f"   [WARN] Datei nicht lesbar für Timerange ({os.path.basename(path)}): {e}"
        )
        return (pd.Timestamp(1900, 1, 1), pd.Timestamp(2100, 1, 1))

    # 1) Bevorzugt: Index-Timestamps
    try:
        ts = pd.to_datetime(group["BZ26.SET.Active Index.Timestamp"].data)
        return (ts.min(), ts.max())
    except KeyError:
        pass

    # 2) Fallback: irgendein .Timestamp-Kanal im PROCESSIMAGE
    tmins, tmaxs = [], []
    for ch in group.channels():
        if ch.name.endswith(".Timestamp"):
            arr = ch.data
            if arr is None or len(arr) == 0:
                continue
            t = pd.to_datetime(arr)
            if len(t) > 0:
                tmins.append(t.min())
                tmaxs.append(t.max())

    if tmins and tmaxs:
        return (min(tmins), max(tmaxs))

    # 3) Harte Fallbacks – lieber nicht ausschließen
    return (pd.Timestamp(1900, 1, 1), pd.Timestamp(2100, 1, 1))


def extract_index_series_all(files):
    all_dfs = []
    skipped = 0
    for f in files:
        try:
            tdms = TdmsFile.read(f)
            group = tdms["PROCESSIMAGE"]
            vals = group["BZ26.SET.Active Index.Current Value"].data
            ts = pd.to_datetime(group["BZ26.SET.Active Index.Timestamp"].data)
            df = pd.DataFrame({"index": vals, "time": ts})
            all_dfs.append(df)
        except KeyError:
            skipped += 1
            continue
    if not all_dfs:
        raise RuntimeError("Kein Indexkanal in irgendeiner Datei gefunden.")
    if skipped:
        print(f"   [INFO] {skipped} Datei(en) ohne Indexkanal übersprungen.")
    return pd.concat(all_dfs).sort_values("time").reset_index(drop=True)


def detect_cycles(index_df):
    cycles = []
    current = []
    seen = set()
    in_cycle = False

    for _, row in index_df.iterrows():
        idx = row["index"]

        if not in_cycle and idx == 99:
            in_cycle = True
            current = [row]
            seen = {idx}
            continue

        if not in_cycle:
            continue

        current.append(row)
        seen.add(idx)

        if idx == 10000:
            df = pd.DataFrame(current)
            cycles.append((df, True))  # unvollständig
            current, seen, in_cycle = [], set(), False
            continue

        if idx == 900:
            df = pd.DataFrame(current)
            # strenge Prüfung: alle REQUIRED_INDICES gesehen?
            unvollst = not REQUIRED_INDICES.issubset(seen)
            cycles.append((df, unvollst))
            current, seen, in_cycle = [], set(), False

    return cycles


def build_channel_pairs(group):
    channels = {ch.name: ch for ch in group.channels()}
    pairs = {}
    for name in channels:
        if name.endswith(".Current Value"):
            base = name[: -len(".Current Value")]
            ts_name = base + ".Timestamp"
            if ts_name in channels:
                values = channels[name].data
                timestamps = pd.to_datetime(channels[ts_name].data)
                pairs[base] = (values, timestamps)
    return pairs


# ========== Kern: Filtern auf Basis von Dateien + Zeitfenstern ==========
def filter_cycle(files, cycle_df, file_ranges=None):
    cycle_df = cycle_df[cycle_df["index"] != 1]
    print("   ↳ Berechne Intervalle...")

    indices = cycle_df["index"].values
    times = pd.to_datetime(cycle_df["time"].values)

    keep_intervals = []
    last_idx = indices[0]
    state_start = times[0]

    for i in range(1, len(indices)):
        idx_now, t_now = indices[i], times[i]

        # Zustand hat gewechselt
        if idx_now != last_idx:
            if last_idx in EXCEPTION_INDICES:
                # Ausnahmezustände: kompletten Abschnitt behalten
                keep_intervals.append((state_start, t_now))
            else:
                # normale Zustände: nur die letzten 30 s
                window_start = max(state_start, t_now - pd.Timedelta(seconds=30))
                keep_intervals.append((window_start, t_now))

            # neuen Zustand beginnen
            state_start = t_now
            last_idx = idx_now

    # --- letztes Intervall behandeln ---
    last_t = times[-1]
    if last_idx in EXCEPTION_INDICES:
        keep_intervals.append((state_start, last_t + pd.Timedelta(seconds=120)))
    else:
        window_start = max(state_start, last_t - pd.Timedelta(seconds=30))
        keep_intervals.append((window_start, last_t))

    # Debug-Ausgabe
    print("   [DEBUG] Keep intervals:")
    for s, e in keep_intervals:
        print(f"      {s} – {e} (Dauer {(e-s).total_seconds():.1f}s)")

    # --- nur relevante Dateien öffnen ---
    cycle_data = {}
    for f in files:
        if file_ranges and (
            file_ranges[f][1] < keep_intervals[0][0]
            or file_ranges[f][0] > keep_intervals[-1][1]
        ):
            continue
        tdms = TdmsFile.read(f)
        for group in tdms.groups():
            if group.name not in cycle_data:
                cycle_data[group.name] = {}
            pairs = build_channel_pairs(group)
            for base, (vals, ts) in pairs.items():
                df = pd.DataFrame({"time": pd.to_datetime(ts), "value": vals})
                mask = pd.Series(False, index=df.index)
                for s, e in keep_intervals:
                    mask |= (df["time"] >= s) & (df["time"] <= e)
                if mask.any():
                    if base not in cycle_data[group.name]:
                        cycle_data[group.name][base] = df.loc[mask]
                    else:
                        cycle_data[group.name][base] = pd.concat(
                            [cycle_data[group.name][base], df.loc[mask]]
                        )

    # sortieren
    for g in cycle_data.values():
        for b in g:
            g[b] = g[b].sort_values("time")

    print("   ✔ Filtercycle abgeschlossen")
    return cycle_data


# ========== Speichern ==========


def save_cycle(
    context, cycle_data, typ, temp, zustand, zyklus_nr, cycle_df, unvollstaendig=False
):
    datum_str = cycle_df["time"].min().strftime("%Y-%m-%d")
    filename = f"{BASENAME}-{typ}_{temp}_{zustand}_{datum_str}_Zyklus{zyklus_nr}"
    if unvollstaendig:
        filename += "_unvollständig"
    filename += ".tdms"
    path = os.path.join(OUTPUT_FOLDER, filename)
    create_archive(
        zyklus_nr,
        typ,
        temp,
        zustand,
        datum_str,
        cycle_data,
        entry_dict={},
        context=context,
        filename=filename,
        file_type="tdms",
        logger=print,
        overwrite=True,
    )
    # with TdmsWriter(path) as writer:
    #     root = RootObject(properties={
    #         "Zyklus": int(zyklus_nr),
    #         "Typ": str(typ),
    #         "Temp": str(temp),
    #         "Zustand": str(zustand),
    #         "Datum": datum_str
    #     })

    #     all_objects = [root]
    #     min_time, max_time, total_points = None, None, 0

    #     for gname, group in cycle_data.items():
    #         gobj = GroupObject(gname)  # properties=None
    #         all_objects.append(gobj)

    #         for cname, df in group.items():
    #             if df.empty:
    #                 continue

    #             # globale Dauer und Punkte zählen
    #             tmin, tmax = df["time"].min(), df["time"].max()
    #             if min_time is None or tmin < min_time:
    #                 min_time = tmin
    #             if max_time is None or tmax > max_time:
    #                 max_time = tmax
    #             total_points += len(df)

    #             values = df["value"].values
    #             ts = df["time"].astype("datetime64[ns]").values
    #             ch_val = ChannelObject(gname, f"{cname}.Current Value", values)
    #             ch_ts  = ChannelObject(gname, f"{cname}.Timestamp", ts)
    #             all_objects.extend([ch_val, ch_ts])

    #     writer.write_segment(all_objects)

    print(f"✔ Datei gespeichert: {path}")
