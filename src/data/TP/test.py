from environment import Environment, Scene, Node, derivative_of
import dill
import pandas as pd
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt

sys.path.append("./src")

np.random.seed(123)

desired_max_time = 100
pred_indices = [2, 3]
state_dim = 6
frame_diff = 10
desired_frame_diff = 1
target_dt = 0.4

standardization = {
    "PEDESTRIAN": {
        "position": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
        "velocity": {"x": {"mean": 0, "std": 2}, "y": {"mean": 0, "std": 2}},
        "acceleration": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
    }
}


def maybe_makedirs(path_to_create):
    """This function will create a directory, unless it exists already,
    at which point the function will return.
    The exception handling is necessary as it prevents a race condition
    from occurring.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise


def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns = pd.MultiIndex.from_product(
        [["position", "velocity", "acceleration"], ["x", "y"]]
    )

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name)

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        x = node.data.position.x.copy()
        y = node.data.position.y.copy()

        x, y = rotate_pc(np.array([x, y]), alpha)

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        data_dict = {
            ("position", "x"): x,
            ("position", "y"): y,
            ("velocity", "x"): vx,
            ("velocity", "y"): vy,
            ("acceleration", "x"): ax,
            ("acceleration", "y"): ay,
        }

        node_data = pd.DataFrame(data_dict, columns=data_columns)

        node = Node(
            node_type=node.type,
            node_id=node.id,
            data=node_data,
            first_timestep=node.first_timestep,
        )

        scene_aug.nodes.append(node)
    return scene_aug


def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug


data_folder_name = "./processed_data/"
maybe_makedirs(data_folder_name)

data_columns = pd.MultiIndex.from_product(
    [["position", "velocity", "acceleration"], ["x", "y"]]
)

nl = 0
l = 0

raw_path = "./raw_data/MOT17Labels/"
leave_out = "MOT17-09-DPM/"

# CONVERTS MOT INTO USABLE DATA FOR MODEL TRAINING
for data_class in ["train", "test", "val"]:
    inputs = []
    for desired_source in os.listdir(os.path.join(raw_path, data_class)):
        frame_rate = 0
        with open(
            os.path.join(raw_path, data_class, desired_source, "seqinfo.ini")
        ) as f1:
            lines = f1.readlines()
            for line in lines:
                if "frameRate" in line:
                    frame_rate = int(line.split("=")[1].strip())
        if data_class == "train" or data_class == "val":
            gt_file = pd.read_csv(
                os.path.join(raw_path, data_class, desired_source, "gt", "gt.txt"),
                usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                header=None,
                names=[
                    "frame_id",
                    "track_id",
                    "bb_left",
                    "bb_top",
                    "bb_width",
                    "bb_height",
                    "is_active",
                    "class_id",
                    "visibility",
                ],
                sep=",",
            )
        else:
            gt_file = pd.read_csv(
                os.path.join(raw_path, data_class, desired_source, "det", "det.txt"),
                usecols=[0, 1, 2, 3, 4, 5, 6],
                header=None,
                names=[
                    "frame_id",
                    "track_id",
                    "bb_left",
                    "bb_top",
                    "bb_width",
                    "bb_height",
                    "conf_score",
                ],
                sep=",",
            )
        inputs.append((gt_file, frame_rate, desired_source))

    h_desired_source = "MOT"
    env = Environment(node_type_list=["PEDESTRIAN"], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
    env.attention_radius = attention_radius

    scenes = []
    data_dict_path = os.path.join(
        data_folder_name, "_".join([h_desired_source, data_class]) + ".pkl"
    )

    for data, frame_rate, desired_source in inputs:
        print(
            f"Processing {data_class} - {desired_source}, with a framerate of {
                frame_rate
            }\n"
        )
        if "class_id" in data.columns:
            data = data[data["class_id"] == 1].copy()
        if data_class == "train" and "is_active" in data.columns:
            data = data[data["is_active"] == 1].copy()
        if data.empty:
            print("No valid data")
            continue
        data["frame_id"] = pd.to_numeric(data["frame_id"], downcast="integer")
        data["track_id"] = pd.to_numeric(data["track_id"], downcast="integer")

        data["x"] = data["bb_left"] + data["bb_width"] / 2.0
        data["y"] = data["bb_top"] + data["bb_height"] / 2.0

        # apply data scale as same as PECnet
        data["x"] = data["x"] / 50
        data["y"] = data["y"] / 50

        # data['frame_id'] -= data['frame_id'].min()

        data["node_type"] = "PEDESTRIAN"
        data["node_id"] = desired_source + "_" + data["track_id"].astype(str)

        data.sort_values("frame_id", inplace=True)

        max_timesteps = data["frame_id"].max()

        if len(data) > 0:
            scene = Scene(
                timesteps=max_timesteps + 1,
                dt=target_dt,
                name=desired_source,
                aug_func=augment if data_class == "train" else None,
            )
            n = 0
            for node_id in pd.unique(data["node_id"]):
                # Use .copy() to avoid SettingWithCopyWarning
                node_df = data[data["node_id"] == node_id].copy()

                # **Add this section to handle duplicates and ensure sort order for the specific node**
                if not node_df.empty:
                    # Sort by frame_id first (important if original sort of 'data' wasn't stable for duplicates)
                    # then drop duplicates, keeping the first entry for that frame.
                    node_df.sort_values("frame_id", inplace=True)
                    node_df.drop_duplicates(
                        subset=["frame_id"], keep="first", inplace=True
                    )

                if len(node_df) > 1:
                    # ... (your debugging pdb.set_trace() and assert logic)
                    frame_differences = np.diff(node_df["frame_id"])
                    is_sequential = np.all(frame_differences == 1)

                    if not is_sequential:
                        print(
                            f"Debugging: Problem detected with node_id: {
                                node_id
                            } AFTER drop_duplicates"
                        )
                        # .to_list() or .values for cleaner print
                        print(f"Frame IDs: {node_df['frame_id'].to_list()}")
                        print(f"Differences: {frame_differences}")
                        import pdb

                        pdb.set_trace()

                    assert (
                        is_sequential
                    ), f"Frame IDs are not sequential for node {
                        node_id
                    } after deduplication"
                    # ... rest of your node processing

                    node_values = node_df[["x", "y"]].values

                    if node_values.shape[0] < 2:
                        continue

                    new_first_idx = node_df["frame_id"].iloc[0]

                    x = node_values[:, 0]
                    y = node_values[:, 1]
                    vx = derivative_of(x, scene.dt)
                    vy = derivative_of(y, scene.dt)
                    ax = derivative_of(vx, scene.dt)
                    ay = derivative_of(vy, scene.dt)

                    data_dict = {
                        ("position", "x"): x,
                        ("position", "y"): y,
                        ("velocity", "x"): vx,
                        ("velocity", "y"): vy,
                        ("acceleration", "x"): ax,
                        ("acceleration", "y"): ay,
                    }

                    node_data = pd.DataFrame(data_dict, columns=data_columns)
                    node = Node(
                        node_type=env.NodeType.PEDESTRIAN,
                        node_id=node_id,
                        data=node_data,
                    )
                    node.first_timestep = new_first_idx

                    scene.nodes.append(node)
            if data_class == "train":
                scene.augmented = list()
                angles = np.arange(0, 360, 15) if data_class == "train" else [0]
                for angle in angles:
                    scene.augmented.append(augment_scene(scene, angle))

            print(scene)
            scenes.append(scene)
    env.scenes = scenes

    if len(scenes) > 0:
        with open(data_dict_path, "wb") as f:
            # pdb.set_trace()
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
