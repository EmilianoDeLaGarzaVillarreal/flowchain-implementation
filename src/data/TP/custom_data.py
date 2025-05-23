from environment import Environment, Scene, Node, derivative_of
import dill
import pickle
import pandas as pd
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
import mot_preprocess as mt

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
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                     [np.sin(alpha), np.cos(alpha)]])
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

leave_out = "MOT17-09-DPM/"


def preprocess():
    raw_path = "./raw_data/MOT17Labels/"
    inputs_train = []
    inputs_test = []
    dt = target_dt
    for desired_source in os.listdir(os.path.join(raw_path, "train")):
        frame_rate = 0
        with open(os.path.join(raw_path, "train", desired_source, "seqinfo.ini")) as f1:
            lines = f1.readlines()
            for line in lines:
                if "frameRate" in line:
                    frame_rate = int(line.split("=")[1].strip())
        path = os.path.join(raw_path, "train", desired_source, "gt", "gt.txt")
        gt_file = mt.create_df(path, dt, frame_rate)
        gt_file = mt.build_meta(gt_file)
        gt_file = mt.rename_and_sort(gt_file)
        gt_file["trackIdWithSource"] = (
            gt_file["trackId"].astype(str) + "_" + desired_source
        )
        standing, moving = mt.identify_tracks(gt_file, 10)
        gt_file_filled = mt.fill_track_gaps(gt_file)
        gt_trimmed = mt.trim_and_filter(gt_file_filled, standing, moving, 9)
        gt_train, gt_test = mt.split_objects(gt_trimmed, 0.3, 42)
        inputs_train.append(gt_train)
        inputs_test.append(gt_test)
        # import pdb
        # pdb.set_trace()
    train_df = pd.concat(inputs_train, ignore_index=True)

    test_df = pd.concat(inputs_test, ignore_index=True)

    with open("./raw_data/MOT/train_trajnet.pkl", "wb") as f:
        pickle.dump(train_df, f)
    with open("./raw_data/MOT/test_trajnet.pkl", "wb") as f:
        pickle.dump(test_df, f)


# CONVERTS MOT INTO USABLE DATA FOR MODEL TRAINING
for data_class in ["train", "test"]:
    preprocess()
    raw_path = "./raw_data/MOT/"
    desired_source = "MOT"
    data_path = os.path.join(raw_path, f"{data_class}_trajnet.pkl")
    print(f"Processing {desired_source.upper()} {data_class}")
    data_out_path = os.path.join(
        data_folder_name, f"{desired_source}_{data_class}.pkl")
    df = pickle.load(open(data_path, "rb"))
    env = Environment(
        node_type_list=["PEDESTRIAN"], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
    env.attention_radius = attention_radius

    scenes = []

    group = df.groupby("sceneId")
    for scene, data in group:
        data["frame"] = pd.to_numeric(data["frame"], downcast="integer")
        data["trackId"] = pd.to_numeric(data["trackId"], downcast="integer")
        data["frame"] = data["frame"] // 3  # CAN CHANGE
        data["combo"] = (
            data["frame"].astype(str) + "_" +
            data["trackIdWithSource"].astype(str)
        )
        data = data.drop_duplicates(subset=["combo"], keep="first").copy()

        # data['frame'] -= data['frame'].min()

        data["node_type"] = "PEDESTRIAN"
        data["node_id"] = data["trackIdWithSource"].astype(str)

        # apply data scale as same as PECnet
        data["x"] = data["x"] / 50
        data["y"] = data["y"] / 50

        # Mean Position
        # data['x'] = data['x'] - data['x'].mean()
        # data['y'] = data['y'] - data['y'].mean()

        max_timesteps = data["frame"].max()

        if len(data) > 0:
            scene = Scene(
                timesteps=max_timesteps + 1,
                dt=target_dt,
                name=scene,
                aug_func=augment if data_class == "train" else None,
            )
            n = 0
            for node_id in pd.unique(data["node_id"]):
                node_df = data[data["node_id"] == node_id]

                if len(node_df) > 1:
                    # assert np.all(np.diff(node_df['frame']) == 1)
                    # import pdb
                    #
                    # pdb.set_trace()
                    if not np.all(np.diff(node_df["frame"]) == 1):
                        import pdb

                        pdb.set_trace()

                    node_values = node_df[["x", "y"]].values

                    if node_values.shape[0] < 2:
                        continue

                    new_first_idx = node_df["frame"].iloc[0]

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
                angles = np.arange(
                    0, 360, 15) if data_class == "train" else [0]
                for angle in angles:
                    scene.augmented.append(augment_scene(scene, angle))

            print(scene)
            scenes.append(scene)
    env.scenes = scenes

    if len(scenes) > 0:
        with open(data_out_path, "wb") as f:
            # pdb.set_trace()
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
