import os
import time
from typing import List, Dict
from yacs.config import CfgNode
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
from collections import OrderedDict

from data.unified_loader import unified_loader
from models.build_model import Build_Model
from metrics.build_metrics import Build_Metrics
from visualization.build_visualizer import Build_Visualizer

import torch.nn as nn
from torch.utils.data import DataLoader


DEVICE = torch.device("cuda")
NUM_CLIENTS = 1

# Here I load the unified loader as a client loader, I pass clied_id and num_clients to get the partition from the dataset and we return
# a DataLoader object


def federatedTrain(
    cfg: CfgNode,
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs=1, #CAN CHANGE WHEN NECESSARY
    proximal_mu=0.0,
    global_model_weights=None,
):
    model.to(DEVICE)
    model.train()

    print(f"[Client] Start local training for {num_epochs} epoch(s)")

    for epoch in range(num_epochs):
        loss_list = []
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=False,
        )
        for batch_idx, data_dict in enumerate(pbar):
            data_dict = {
                k: (
                    data_dict[k].cuda()
                    if isinstance(data_dict[k], torch.Tensor)
                    else data_dict[k]
                )
                for k in data_dict
            }
            if proximal_mu > 0:
                loss_info = model.update_with_mu_values(data_dict, proximal_mu, global_model_weights)
            else:
                loss_info = model.update(data_dict)
            loss_list.append(loss_info)
            pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_info.items()})

    loss_info = aggregate(loss_list)
    print(
        f"[Client] Finished local training | Loss: {
          loss_info.get('loss', 0.0):.4f}"
    )
    return loss_info.get("loss", 0.0)


def federatedTest(cfg: CfgNode, model: nn.Module, round, visualize):
    data_loader = unified_loader(cfg, rand=False, split="test")
    result_info = evaluate_model(cfg, model, data_loader, visualize, round)
    print("RESULT INFO:", result_info)
    import json

    def convert_to_json_serializable(obj):
        if isinstance(obj, dict):
            # If it's a dictionary, apply this function to each of its values.
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # If it's a list, apply this function to each of its elements.
            return [convert_to_json_serializable(elem) for elem in obj]
        elif isinstance(obj, np.ndarray):
            # If it's a NumPy array (like your 'log_prob'), convert it to a Python list.
            return obj.tolist()
        # Handles 'score', 'ade', 'fde', etc.
        elif isinstance(obj, np.float64):
            return float(obj)  # Convert NumPy float64 to Python float.
        elif isinstance(obj, np.int64):  # Add other np integer types if you have them
            return int(obj)  # Convert NumPy int64 to Python int.
        elif isinstance(obj, np.bool_):
            return bool(obj)  # Convert NumPy bool to Python bool.
        # A more general catch-all for other NumPy scalar types
        elif isinstance(obj, np.generic):
            return (
                obj.item()
                # Converts to the closest Python native type (e.g. np.float32 -> float)
            )
        # If it's already a standard Python type, return it as is.
        return obj

    serialized_result = convert_to_json_serializable(result_info)

    with open(os.path.join(cfg.OUTPUT_DIR, "metrics.json"), "w") as fp:
        json.dump(serialized_result, fp)
    return result_info


def evaluate_model(
    cfg: CfgNode,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    visualize=False,
    round=0,
):
    model.eval()
    metrics = Build_Metrics(cfg)
    visualizer = Build_Visualizer(cfg)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    update_timesteps = [1]

    run_times = {0: []}
    run_times.update({t: [] for t in update_timesteps})

    result_info = {}

    if visualize and round == 10:
        with torch.no_grad():
            result_list = []
            print(
                "timing the computation, evaluating probability map, and visualizing... "
            )
            data_loader_one_each = unified_loader(
                cfg, rand=False, split="test", batch_size=10
            )
            for i, data_dict in enumerate(
                tqdm(data_loader_one_each, leave=False, total=1)
            ):
                data_dict = {
                    k: (
                        data_dict[k].cuda()
                        if isinstance(data_dict[k], torch.Tensor)
                        else data_dict[k]
                    )
                    for k in data_dict
                }
                dict_list = []
                result_dict = model.predict(
                    deepcopy(data_dict), return_prob=True
                )  # warm-up
                torch.cuda.synchronize()
                starter.record()
                result_dict = model.predict(deepcopy(data_dict), return_prob=True)
                ender.record()
                torch.cuda.synchronize()
                curr_run_time = starter.elapsed_time(ender)
                run_times[0].append(curr_run_time)

                for t in update_timesteps:
                    starter.record()
                    result_dict = model.predict_from_new_obs(result_dict, t)
                    ender.record()
                    torch.cuda.synchronize()
                    curr_run_time = starter.elapsed_time(ender)
                    run_times[t].append(curr_run_time)

                dict_list.append(deepcopy(result_dict))
                dict_list = metrics.denormalize(dict_list)  # denormalize the output
                if cfg.TEST.KDE:
                    torch.cuda.synchronize()
                    starter.record()
                    dict_list = kde(dict_list)
                    ender.record()
                    torch.cuda.synchronize()
                    run_times[0][-1] += starter.elapsed_time(ender)
                dict_list = visualizer.prob_to_grid(dict_list)
                result_list.append(metrics(deepcopy(dict_list)))

                if visualize:
                    visualizer(dict_list)
                if i == 9:
                    break

            result_info.update(aggregate(result_list))

        print(
            f"execution time: {np.mean(run_times[0]):.2f} "
            + "\u00b1"
            + f"{np.std(run_times[0]):.2f} [ms]"
        )
        print(
            f"execution time: {np.mean(run_times[1]):.2f} "
            + "\u00b1"
            + f"{np.std(run_times[1]):.2f} [ms]"
        )
        result_info.update(
            {"execution time": np.mean(run_times[0]), "time std": np.std(run_times[0])}
        )

    print("evaluating ADE/FDE metrics ...")

    inference_times = []
    with torch.no_grad():
        result_list = []
        for i, data_dict in enumerate(tqdm(data_loader, leave=False)):
            data_dict = {
                k: (
                    data_dict[k].cuda()
                    if isinstance(data_dict[k], torch.Tensor)
                    else data_dict[k]
                )
                for k in data_dict
            }

            dict_list = []
            for _ in range(cfg.TEST.N_TRIAL):
                start = time.time()
                result_dict = model.predict(deepcopy(data_dict), return_prob=False)
                end = time.time()
                inference_times.append(end - start)
                dict_list.append(deepcopy(result_dict))

            dict_list = metrics.denormalize(dict_list)
            result_list.append(deepcopy(metrics(dict_list)))
        d = aggregate(result_list)
        result_info.update({k: d[k] for k in d.keys() if d[k] != 0.0})

    # Add average inference time
    if inference_times:
        avg_time = np.mean(inference_times)
        result_info["inference_time"] = avg_time
        print(f"[Client] Avg inference time per prediction: {avg_time:.6f}s")

        # Auto-detect batch size for FPS calculation
        first_key = next(iter(data_dict))
        if isinstance(data_dict[first_key], torch.Tensor):
            batch_size = data_dict[first_key].shape[0]
        else:
            batch_size = 1  # Fallback if not tensor

        fps = batch_size / avg_time if avg_time > 0 else 0.0
        result_info["fps"] = fps
        print(f"[Client] Estimated FPS: {fps:.2f} (Batch size: {batch_size})")

    np.set_printoptions(precision=4)

    model.train()

    return result_info


def aggregate(dict_list: List[Dict]) -> Dict:
    if "nsample" in dict_list[0]:
        ret_dict = {
            k: np.sum([d[k] for d in dict_list], axis=0)
            / np.sum([d["nsample"] for d in dict_list])
            for k in dict_list[0].keys()
        }
    else:
        ret_dict = {
            k: np.mean([d[k] for d in dict_list], axis=0) for k in dict_list[0].keys()
        }

    return ret_dict


def tune(cfg: CfgNode) -> None:
    import optuna

    def objective_with_arg(cfg):
        _cfg = cfg.clone()
        _cfg.defrost()

        def objective(trial):
            _cfg.MODEL.FLOW.N_BLOCKS = trial.suggest_int("MODEL.FLOW.N_BLOCKS", 1, 3)
            _cfg.MODEL.FLOW.N_HIDDEN = trial.suggest_int("MODEL.FLOW.N_HIDDEN", 1, 3)
            _cfg.MODEL.FLOW.HIDDEN_SIZE = trial.suggest_int(
                "MODEL.FLOW.HIDDEN_SIZE", 32, 128, step=16
            )
            _cfg.MODEL.FLOW.CONDITIONING_LENGTH = trial.suggest_int(
                "MODEL.FLOW.CONDITIONING_LENGTH", 8, 64, step=8
            )
            _cfg.SOLVER.LR = trial.suggest_float("SOLVER.LR", 1e-6, 1e-3, log=True)
            _cfg.SOLVER.WEIGHT_DECAY = trial.suggest_float(
                "SOLVER.WEIGHT_DECAY", 1e-12, 1e-5, log=True
            )

            return train(_cfg, save_model=False)

        return objective

    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.HyperbandPruner()

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        storage=os.path.join("sqlite:///", cfg.OUTPUT_DIR, "optuna.db"),
        study_name="my_opt",
        load_if_exists=True,
    )
    study.optimize(objective_with_arg(cfg), n_jobs=4, n_trials=200, gc_after_trial=True)

    trial = study.best_trial

    print(trial.value, trial.params)


def kde(dict_list: List):
    from utils import GaussianKDE

    for data_dict in dict_list:
        for k in list(data_dict.keys()):
            if k[0] == "prob":
                prob = data_dict[k]
                batch_size, _, timesteps, _ = prob.shape
                prob_, gt_traj_log_prob = [], []
                for b in range(batch_size):
                    prob__, gt_traj_prob__ = [], []
                    for i in range(timesteps):
                        kernel = GaussianKDE(prob[b, :, i, :-1])
                        # estimate the prob of predicted future positions for fair comparison of inference time
                        kernel(prob[b, :, i, :-1])
                        prob__.append(deepcopy(kernel))
                        gt_traj_prob__.append(
                            kernel(data_dict["gt"][b, None, i].float())
                        )
                    prob_.append(deepcopy(prob__))
                    gt_traj_log_prob.append(torch.cat(gt_traj_prob__, dim=-1).log())
                gt_traj_log_prob = torch.stack(gt_traj_log_prob, dim=0)
                gt_traj_log_prob = torch.nan_to_num(gt_traj_log_prob, neginf=-10000)
                data_dict[k] = prob_
                data_dict[("gt_traj_log_prob", k[1])] = gt_traj_log_prob

    return dict_list
