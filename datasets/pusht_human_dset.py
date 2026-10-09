"""Push-T datasets built by phone2panda/make_dataset.py.

Identical to datasets/pusht_dset.py except for normalisation: the stock loader hard-codes
DINO-WM's action/state/proprio statistics, while these datasets ship their own, computed
on the training split, in <data_path>/stats.pth.
"""
import torch

from .pusht_dset import PushTDataset
from .traj_dset import TrajSlicerDataset


class PushTHumanDataset(PushTDataset):
    def __init__(self, stats_path, normalize_action=True, **kwargs):
        super().__init__(normalize_action=False, **kwargs)  # raw arrays, identity normalisation
        if normalize_action:
            s = torch.load(stats_path)
            self.action_mean, self.action_std = s["action_mean"], s["action_std"]
            self.state_mean, self.state_std = s["state_mean"][: self.state_dim], s["state_std"][: self.state_dim]
            self.proprio_mean = s["proprio_mean"][: self.proprio_dim]
            self.proprio_std = s["proprio_std"][: self.proprio_dim]
            self.actions = (self.actions - self.action_mean) / self.action_std
            self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std


def load_pusht_human_slice_train_val(
    transform,
    n_rollout=None,
    data_path="data/pusht_human_aug",
    normalize_action=True,
    split_ratio=0.9,  # unused: the split is fixed by the train/ and val/ folders
    num_hist=0,
    num_pred=0,
    frameskip=0,
    with_velocity=True,
    reg_window=None,
):
    common = dict(n_rollout=n_rollout, transform=transform, normalize_action=normalize_action,
                  with_velocity=with_velocity, stats_path=data_path + "/stats.pth")
    train_dset = PushTHumanDataset(data_path=data_path + "/train", **common)
    val_dset = PushTHumanDataset(data_path=data_path + "/val", **common)

    num_frames = num_hist + num_pred
    if reg_window is not None:  # regularizer window longer than the predictor's (train.py reg_window)
        num_frames = max(num_frames, int(reg_window))
    datasets = {
        "train": TrajSlicerDataset(train_dset, num_frames, frameskip),
        "valid": TrajSlicerDataset(val_dset, num_frames, frameskip),
    }
    traj_dset = {"train": train_dset, "valid": val_dset}
    return datasets, traj_dset
