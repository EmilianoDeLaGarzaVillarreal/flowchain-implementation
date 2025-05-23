import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_df(filepath, delta_time, orig_fps):
    cols = ["frame_id", "track_id", "x", "y", "w", "h", "_7", "_8", "_9"]
    df = pd.read_csv(filepath, header=None, names=cols, usecols=cols[:6])

    # Global sub-sampling of frames
    step = int(round(delta_time * orig_fps))
    all_frames = sorted(df["frame_id"].unique())
    sampled = all_frames[::step]
    df = df[df["frame_id"].isin(sampled)].copy()

    # Remap frames to 1...N
    frame_map = {old: new for new, old in enumerate(sampled, start=1)}
    df["frame_id"] = df["frame_id"].map(frame_map)
    return df


def build_meta(df):
    df["sceneId"] = "cuip"
    unique_tracks = sorted(df["track_id"].unique())
    track_to_meta = {t: i for i, t in enumerate(unique_tracks)}
    df["metaId"] = df["track_id"].map(track_to_meta)
    return df


def rename_and_sort(df):
    return (
        df.rename(columns={"frame_id": "frame", "track_id": "trackId"})[
            ["frame", "trackId", "x", "y", "sceneId", "metaId"]
        ]
        .sort_values(["trackId", "frame"])
        .reset_index(drop=True)
    )


def identify_tracks(df, min_movement):
    standing, moving = [], []
    for tid, grp in df.groupby("trackId", sort=False):
        dx = grp["x"].iloc[-1] - grp["x"].iloc[0]
        dy = grp["y"].iloc[-1] - grp["y"].iloc[0]
        if np.hypot(dx, dy) < min_movement:
            standing.append(tid)
        else:
            moving.append(tid)
    return standing, moving


def fill_track_gaps(df):
    out = []
    for tid, grp in df.groupby("trackId", sort=False):
        grp = grp.sort_values("frame").set_index("frame")
        full = np.arange(grp.index.min(), grp.index.max() + 1)
        gf = grp.reindex(full).assign(
            trackId=lambda d: d["trackId"].ffill(),
            sceneId=lambda d: d["sceneId"].ffill(),
            metaId=lambda d: d["metaId"].ffill(),
        )
        gf[["x", "y"]] = gf[["x", "y"]].interpolate().ffill().bfill()
        gf = gf.reset_index().rename(columns={"index": "frame"})
        gf["trackId"] = gf["trackId"].astype(int)
        gf["metaId"] = gf["metaId"].astype(int)
        out.append(gf)
    return pd.concat(out, ignore_index=True)


def trim_and_filter(df_filled, standing_ids, moving_ids, min_frames):
    # Trim standing to max 20 frames
    stand = (
        df_filled[df_filled["trackId"].isin(standing_ids)]
        .groupby("trackId", group_keys=False)
        .head(20)
    )
    move = df_filled[df_filled["trackId"].isin(moving_ids)]
    # df = pd.concat([stand, move], ignore_index=True)
    df = move.copy()

    # Drop short tracks
    if min_frames is not None:
        counts = df["trackId"].value_counts()
        valid = counts[counts >= min_frames].index
        df = df[df["trackId"].isin(valid)].reset_index(drop=True)
    return df


def split_objects(df, test_size, random_state):
    track_ids = sorted(df["trackId"].unique())
    train_ids, test_ids = train_test_split(
        track_ids, test_size=test_size, random_state=random_state, shuffle=True
    )
    train = df[df["trackId"].isin(train_ids)].reset_index(drop=True)
    test = df[df["trackId"].isin(test_ids)].reset_index(drop=True)
    return train, test
