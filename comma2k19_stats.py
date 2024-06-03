import pickle
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import Comma2k19SequenceDataset


def calculate_speed(waypoints: torch.Tensor, time_offsets: np.ndarray):
    start_waypoint = waypoints[0]
    middle_waypoint = waypoints[3]
    # end_waypoint = waypoints[-1]
    dist_middle = np.linalg.norm(middle_waypoint - start_waypoint)
    # dist_end = np.linalg.norm(end_waypoint - middle_waypoint)
    middle_speed = dist_middle / time_offsets[3]
    return middle_speed


def fit_circle(points):
    """
    Fit a circle to a set of 2D points.
    """

    def calc_R(xc, yc):
        """Calculate the distance of each 2D point from the center (xc, yc)."""
        return np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2)

    def cost(params):
        """Calculate the algebraic distance between the data points and the mean circle centered at (xc, yc)."""
        Ri = calc_R(*params)
        return np.std(Ri)

    x_m = np.mean(points[:, 0])
    y_m = np.mean(points[:, 1])
    center_estimate = x_m, y_m
    result = minimize(cost, center_estimate)
    xc, yc = result.x
    Ri = calc_R(xc, yc)
    R = np.mean(Ri)

    return xc, yc, R


def calculate_curvature(waypoints):
    """
    Calculate the curvature of the trajectory described by the waypoints.

    Args:
        waypoints (np.ndarray): An array of shape (N, 2) representing the x and y coordinates of the waypoints.

    Returns:
        float: The curvature of the trajectory.
    """
    if waypoints.shape[0] < 3:
        raise ValueError("At least 3 waypoints are required to fit a circle.")

    xc, yc, R = fit_circle(waypoints)
    curvature = 1 / R if R != 0 else float("inf")

    return curvature


def calculate_yaw(waypoints):
    """
    Calculate the yaw angle of the trajectory described by the waypoints.

    Args:
        waypoints (np.ndarray): An array of shape (N, 2) representing the x and y coordinates of the waypoints.

    Returns:
        float: The yaw angle of the trajectory.
    """
    if waypoints.shape[0] < 2:
        raise ValueError(
            "At least 2 waypoints are required to calculate the yaw angle."
        )

    start_waypoint = waypoints[0]
    end_waypoint = waypoints[3]
    yaw = np.arctan2(
        end_waypoint[1] - start_waypoint[1], end_waypoint[0] - start_waypoint[0]
    )
    return yaw


def calculate_distance(waypoints):
    """
    Calculate the distance of the trajectory described by the waypoints.

    Args:
        waypoints (np.ndarray): An array of shape (N, 2) representing the x and y coordinates of the waypoints.

    Returns:
        float: The distance of the trajectory.
    """

    start_waypoint = waypoints[0]
    end_waypoint = waypoints[3]
    distance = np.linalg.norm(end_waypoint - start_waypoint)
    return distance


@click.command()
@click.option("--folder", default="data/", help="Folder to comma base")
def main(folder: str):
    stats_folder = Path("./stats") / folder
    stats_folder.mkdir(exist_ok=True, parents=True)
    print("Using folder", folder)
    base_folder = Path(folder)
    comma_folder = base_folder / "comma2k19"
    file = base_folder / "comma2k19_val_non_overlap.txt"
    data = Comma2k19SequenceDataset(
        file.as_posix(),
        comma_folder.as_posix() + "/",
        "train",
        use_memcache=False,
        return_origin=False,
    )
    dataloader = DataLoader(data, 1, num_workers=0, shuffle=False)

    seq_idx = 0
    for b_idx, batch in enumerate(dataloader):
        seq_inputs, seq_labels = (
            batch["seq_input_img"].cpu(),
            batch["seq_future_poses"].cpu(),
        )
        seq_length = seq_labels.size(1)
        stats = {"speeds": [], "curvatures": [], "yaws": [], "distance": []}

        # hidden = torch.zeros((2, bs, 512), device=seq_inputs.device)

        for t in range(seq_length):

            inputs, labels = seq_inputs[:, t, :, :, :], seq_labels[:, t, :, :]
            # pred_cls, pred_trajectory, hidden = planning_v0(inputs, hidden)

            # pred_conf = softmax(pred_cls, dim=-1).cpu().numpy()[0]

            inputs, labels = inputs.cpu().numpy()[0], labels.cpu().numpy()[0]
            stats["speeds"].append(calculate_speed(labels, data.t_anchors))
            stats["curvatures"].append(calculate_curvature(labels))
            stats["yaws"].append(calculate_yaw(labels))
            stats["distance"].append(calculate_distance(labels))
        time = datetime.now().strftime("%d-%H-%M")
        name = f"{time}_{b_idx}"
        with open(f"stats/{name}_dict.pkl", "wb") as f:
            pickle.dump(stats, f)
        for key, value in stats.items():
            plt.plot(value)
            plt.title(key)
            plt.savefig((stats_folder / f"{name}_{key}.png").as_posix())
            plt.close()
        print("Saved stats for batch", b_idx, "of ", len(dataloader))

        seq_idx += 1


if __name__ == "__main__":
    main()
