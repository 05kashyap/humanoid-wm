import os
import warnings

# Quiet the import-time noise of gym, d4rl, pybullet and hydra ("Gym has been unmaintained",
# "Overriding environment maze2d-...", D4RL import errors, deprecation warnings). Set through
# environment variables so the spawned env worker processes start quiet too.
# `PYTHONWARNINGS=default python plan.py ...` brings the warnings back.
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
if os.environ["PYTHONWARNINGS"] == "ignore":
    warnings.filterwarnings("ignore")
    import contextlib, io
    with contextlib.redirect_stderr(io.StringIO()):
        import gym  # prints the "Gym has been unmaintained" notice on import
else:
    import gym
import json
import time
import hydra
import random
import torch
import pickle
import wandb
import logging
import numpy as np
import submitit
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import cfg_to_dict, seed

log = logging.getLogger(__name__)


ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]

def planning_main_in_dir(working_dir, cfg_dict):
    os.chdir(working_dir)
    return planning_main(cfg_dict=cfg_dict)

def launch_plan_jobs(
    epoch,
    cfg_dicts,
    plan_output_dir,
):
    with submitit.helpers.clean_env():
        jobs = []
        for cfg_dict in cfg_dicts:
            subdir_name = f"{cfg_dict['planner']['name']}_goal_source={cfg_dict['goal_source']}_goal_H={cfg_dict['goal_H']}_alpha={cfg_dict['objective']['alpha']}"
            subdir_path = os.path.join(plan_output_dir, subdir_name)
            executor = submitit.AutoExecutor(
                folder=subdir_path, slurm_max_num_timeout=20
            )
            executor.update_parameters(
                **{
                    k: v
                    for k, v in cfg_dict["hydra"]["launcher"].items()
                    if k != "submitit_folder"
                }
            )
            cfg_dict["saved_folder"] = subdir_path
            cfg_dict["wandb_logging"] = False  # don't init wandb
            job = executor.submit(planning_main_in_dir, subdir_path, cfg_dict)
            jobs.append((epoch, subdir_name, job))
            print(
                f"Submitted evaluation job for checkpoint: {subdir_path}, job id: {job.job_id}"
            )
        return jobs


def build_plan_cfg_dicts(
    plan_cfg_path="",
    ckpt_base_path="",
    model_name="",
    model_epoch="final",
    planner=["gd", "cem"],
    goal_source=["dset"],
    goal_H=[1, 5, 10],
    alpha=[0, 0.1, 1],
):
    """
    Return a list of plan overrides, for model_path, add a key in the dict {"model_path": model_path}.
    """
    config_path = os.path.dirname(plan_cfg_path)
    overrides = [
        {
            "planner": p,
            "goal_source": g_source,
            "goal_H": g_H,
            "ckpt_base_path": ckpt_base_path,
            "model_name": model_name,
            "model_epoch": model_epoch,
            "objective": {"alpha": a},
        }
        for p, g_source, g_H, a in product(planner, goal_source, goal_H, alpha)
    ]
    cfg = OmegaConf.load(plan_cfg_path)
    cfg_dicts = []
    for override_args in overrides:
        planner = override_args["planner"]
        planner_cfg = OmegaConf.load(
            os.path.join(config_path, f"planner/{planner}.yaml")
        )
        cfg["planner"] = OmegaConf.merge(cfg.get("planner", {}), planner_cfg)
        override_args.pop("planner")
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_args))
        cfg_dict = OmegaConf.to_container(cfg)
        cfg_dict["planner"]["horizon"] = cfg_dict["goal_H"]  # assume planning horizon equals to goal horizon
        cfg_dicts.append(cfg_dict)
    return cfg_dicts


class PlanWorkspace:
    def __init__(
        self,
        cfg_dict: dict,
        wm: torch.nn.Module,
        dset,
        env: SubprocVectorEnv,
        env_name: str,
        frameskip: int,
        wandb_run: wandb.run,
    ):
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.device = next(wm.parameters()).device

        # have different seeds for each planning instances
        self.eval_seed = [cfg_dict["seed"] * n + 1 for n in range(cfg_dict["n_evals"])]
        print("eval_seed: ", self.eval_seed)
        self.n_evals = cfg_dict["n_evals"]
        self.chunk_size = max(
            1, min(int(cfg_dict.get("chunk_size") or self.n_evals), self.n_evals)
        )
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = cfg_dict["debug_dset_init"]

        objective_fn = hydra.utils.call(
            cfg_dict["objective"],
        )

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        if self.cfg_dict["goal_source"] == "file":
            self.prepare_targets_from_file(cfg_dict["goal_file_path"])
        else:
            self.prepare_targets()

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
            decode_for_viz=self.cfg_dict.get("decode_for_viz", True),
            chunk_size=self.chunk_size,
        )

        if self.wandb_run is None or isinstance(
            self.wandb_run, wandb.sdk.lib.disabled.RunDisabled
        ):
            self.wandb_run = DummyWandbRun()

        self.goal_H_model_steps = self.goal_H // self.frameskip
        self.log_filename = "logs.json"  # planner and final eval logs are dumped here
        planner_cfg = self.cfg_dict["planner"].copy()
        if "n_taken_actions" in planner_cfg:
            planner_cfg["n_taken_actions"] = planner_cfg["n_taken_actions"] // self.frameskip
        if "sub_planner" in planner_cfg and "horizon" in planner_cfg["sub_planner"]:
            planner_cfg["sub_planner"]["horizon"] = (
                planner_cfg["sub_planner"]["horizon"] // self.frameskip
            )
        elif "horizon" in planner_cfg:
            planner_cfg["horizon"] = planner_cfg["horizon"] // self.frameskip
        self.planner = hydra.utils.instantiate(
            planner_cfg,
            wm=self.wm,
            env=self.env,  # only for mpc
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
        )


        self.planner.horizon = self.goal_H_model_steps

        self.dump_targets()

    def prepare_targets(self):
        states = []
        actions = []
        observations = []
        
        if self.goal_source == "random_state":
            # update env config from val trajs (chunked to match env workers)
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=2)
            )
            self.env_info = env_info

            # sample random states (chunked to match env workers)
            rand_init_list, rand_goal_list = [], []
            for start in range(0, self.n_evals, self.chunk_size):
                end = min(start + self.chunk_size, self.n_evals)
                self.env.update_env(env_info[start:end])
                ri, rg = self.env.sample_random_init_goal_states(
                    self.eval_seed[start:end]
                )
                rand_init_list.append(ri)
                rand_goal_list.append(rg)
            rand_init_state = np.concatenate(rand_init_list, axis=0)
            rand_goal_state = np.concatenate(rand_goal_list, axis=0)
            if self.env_name == "deformable_env": # take rand init state from dset for deformable envs
                rand_init_state = np.array([x[0] for x in states])

            obs_0, state_0 = self._chunked_env_prepare(self.eval_seed, rand_init_state)
            obs_g, state_g = self._chunked_env_prepare(self.eval_seed, rand_goal_state)

            # add dim for t
            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = rand_init_state  # (b, d)
            self.state_g = rand_goal_state
            self.gt_actions = None
        else:
            # update env config from val trajs (chunked to match env workers)
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=self.goal_H + 1)
            )
            self.env_info = env_info

            # get states from val trajs
            init_state = [x[0] for x in states]
            init_state = np.array(init_state)
            actions = torch.stack(actions)
            if self.goal_source == "random_action":
                actions = torch.randn_like(actions)
            wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
            exec_actions = self.data_preprocessor.denormalize_actions(actions)
            # replay actions in env to get gt obses (chunked to match env workers)
            rollout_obses_list, rollout_states_list = [], []
            for start in range(0, self.n_evals, self.chunk_size):
                end = min(start + self.chunk_size, self.n_evals)
                self.env.update_env(env_info[start:end])
                o, s = self.env.rollout(
                    self.eval_seed[start:end],
                    init_state[start:end],
                    exec_actions.numpy()[start:end],
                )
                # keep only the first and last frames: whole rollouts of n_evals episodes
                # (goal_H + 1 images each) do not fit in a laptop's RAM for n_evals=200
                rollout_obses_list.append({k: v[:, [0, -1]].copy() for k, v in o.items()})
                rollout_states_list.append(s)
                del o
            rollout_obses = {
                key: np.concatenate([d[key] for d in rollout_obses_list], axis=0)
                for key in rollout_obses_list[0]
            }
            rollout_states = np.concatenate(rollout_states_list, axis=0)
            self.obs_0 = {key: arr[:, :1] for key, arr in rollout_obses.items()}
            self.obs_g = {key: arr[:, 1:] for key, arr in rollout_obses.items()}
            self.state_0 = init_state  # (b, d)
            self.state_g = rollout_states[:, -1]  # (b, d)
            self.gt_actions = wm_actions

    def _chunked_env_prepare(self, seeds, init_states):
        """env.prepare over chunks so the batch always matches the env's worker count."""
        obs_list, state_list = [], []
        for start in range(0, self.n_evals, self.chunk_size):
            end = min(start + self.chunk_size, self.n_evals)
            self.env.update_env(self.env_info[start:end])
            o, s = self.env.prepare(seeds[start:end], init_states[start:end])
            obs_list.append(o)
            state_list.append(s)
        obs = {
            k: np.concatenate([d[k] for d in obs_list], axis=0)
            for k in obs_list[0]
        }
        state = np.concatenate(state_list, axis=0)
        return obs, state

    def sample_traj_segment_from_dset(self, traj_len):
        states = []
        actions = []
        observations = []
        env_info = []

        # Check if any trajectory is long enough
        valid_traj = [
            self.dset[i][0]["visual"].shape[0]
            for i in range(len(self.dset))
            if self.dset[i][0]["visual"].shape[0] >= traj_len
        ]
        if len(valid_traj) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")

        # sample init_states from dset
        for i in range(self.n_evals):
            max_offset = -1
            while max_offset < 0:  # filter out traj that are not long enough
                traj_id = random.randint(0, len(self.dset) - 1)
                obs, act, state, e_info = self.dset[traj_id]
                max_offset = obs["visual"].shape[0] - traj_len
            state = state.numpy()
            offset = random.randint(0, max_offset)
            # the sampled frames are not used (the targets are re-rendered by the env), and
            # keeping slices of them held every sampled demo's whole video in memory
            del obs
            state = state[offset : offset + traj_len]
            act = act[offset : offset + self.goal_H]
            actions.append(act)
            states.append(state)
            env_info.append(e_info)
        return observations, states, actions, env_info

    def prepare_targets_from_file(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.obs_0 = data["obs_0"]
        self.obs_g = data["obs_g"]
        self.state_0 = data["state_0"]
        self.state_g = data["state_g"]
        self.gt_actions = data["gt_actions"]
        self.goal_H = data["goal_H"]
        # a task file can hold more tasks than n_evals: use the first n_evals (the same tasks
        # for every model, so e.g. closed loop on 50 = the first 50 of the open-loop 200)
        n_file = len(self.state_0)
        if n_file < self.n_evals:
            raise ValueError(f"{file_path} holds {n_file} tasks, fewer than n_evals={self.n_evals}")
        if n_file > self.n_evals:
            n = self.n_evals
            self.obs_0 = {k: v[:n] for k, v in self.obs_0.items()}
            self.obs_g = {k: v[:n] for k, v in self.obs_g.items()}
            self.state_0, self.state_g = self.state_0[:n], self.state_g[:n]
            if self.gt_actions is not None:
                self.gt_actions = self.gt_actions[:n]
            print(f"[plan.py] using the first {n} of the {n_file} tasks in {file_path}")

    def dump_targets(self):
        with open("plan_targets.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_0": self.obs_0,
                    "obs_g": self.obs_g,
                    "state_0": self.state_0,
                    "state_g": self.state_g,
                    "gt_actions": self.gt_actions,
                    "goal_H": self.goal_H,
                },
                f,
            )
        file_path = os.path.abspath("plan_targets.pkl")
        print(f"Dumped plan targets to {file_path}")

    def perform_planning(self):
        # Plan in chunks of self.chunk_size so the GPU/CPU memory of each
        # planner call (and its internal eval_actions calls) stays bounded.
        all_actions, all_action_len = [], []
        for start in range(0, self.n_evals, self.chunk_size):
            end = min(start + self.chunk_size, self.n_evals)
            chunk_obs_0 = {k: v[start:end] for k, v in self.obs_0.items()}
            chunk_obs_g = {k: v[start:end] for k, v in self.obs_g.items()}
            chunk_state_0 = self.state_0[start:end]
            chunk_state_g = self.state_g[start:end]
            # point the shared evaluator at the current chunk (planners call
            # eval_actions internally, e.g. CEM eval_every / MPC per-iter).
            self.evaluator.task_offset = start  # videos of this chunk are named by task number
            self.evaluator.assign_init_cond(obs_0=chunk_obs_0, state_0=chunk_state_0)
            self.evaluator.assign_goal_cond(obs_g=chunk_obs_g, state_g=chunk_state_g)
            if self.debug_dset_init:
                actions_init = self.gt_actions[start:end]
            else:
                actions_init = None
            actions, action_len = self.planner.plan(
                obs_0=chunk_obs_0,
                obs_g=chunk_obs_g,
                actions=actions_init,
            )
            all_actions.append(actions)
            all_action_len.append(action_len)
            # read by run_scripts/plan_eval.sh for its progress bar
            print(f"[progress] chunk {start // self.chunk_size + 1}/"
                  f"{(self.n_evals + self.chunk_size - 1) // self.chunk_size}", flush=True)
        # Closed-loop MPC returns per-chunk horizons that can differ (each
        # episode succeeds at its own MPC iteration), so pad every chunk to the
        # max horizon before concatenating. eval_actions truncates each episode
        # by its own action_len, so the padded tail is never scored.
        # The padding must be the action "do nothing", which is not 0 in normalized units
        # (0 there is the mean action): padding with 0 made episodes that finished early drift
        # for the rest of the final rollout.
        max_T = max(a.shape[1] for a in all_actions)
        d_raw = self.action_dim // self.frameskip
        still = self.data_preprocessor.normalize_actions(torch.zeros(1, self.frameskip, d_raw))
        still = rearrange(still, "b f d -> b (f d)").to(all_actions[0].device, all_actions[0].dtype)
        actions = torch.cat(
            [
                torch.cat([a, still.expand(a.shape[0], max_T - a.shape[1], -1)], dim=1)
                for a in all_actions
            ],
            dim=0,
        )
        action_len = np.concatenate(all_action_len)

        # final eval over the full episode set (eval_actions chunks internally
        # so the batch always matches the env's workers)
        print("[progress] final evaluation", flush=True)
        self.evaluator.task_offset = 0
        self.evaluator.assign_init_cond(obs_0=self.obs_0, state_0=self.state_0)
        self.evaluator.assign_goal_cond(obs_g=self.obs_g, state_g=self.state_g)
        logs, successes, _, e_states = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=True, filename="output_final"
        )
        # per-episode results, for paired comparisons between models (helpers/plan_report.py)
        e_states = np.asarray(e_states)
        # the state each episode is scored at: where it succeeded (MPC stops there), else the end
        end = np.where(np.isfinite(action_len), action_len * self.frameskip, e_states.shape[1] - 1)
        end = np.clip(end, 0, e_states.shape[1] - 1).astype(int)
        np.savez(
            "final_eval_results.npz",
            success=np.asarray(successes, dtype=bool),
            e_states=e_states,
            final_state=e_states[np.arange(len(end)), end],
            action_len=np.asarray(action_len, dtype=float),
            state_0=np.asarray(self.state_0),
            state_g=np.asarray(self.state_g),
        )
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)
        logs_entry = {
            key: (
                value.item()
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs


def load_ckpt(snapshot_path, device):
    from models.dino import DinoV2Encoder
    _ = DinoV2Encoder('dinov2_vits14', 'x_norm_patchtokens')
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device)
    loaded_keys = []
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result


def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(
            train_cfg.encoder,
        )
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = torch.load(decoder_path)
        else:
            raise ValueError(
                "Decoder path not found in model checkpoint \
                                and is not provided in config"
            )
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    return model


class DummyWandbRun:
    def __init__(self):
        self.mode = "disabled"

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def planning_main(cfg_dict):
    t_start = time.perf_counter()
    t_after_model = None
    t_after_workspace = None
    t_after_planning = None

    output_dir = cfg_dict["saved_folder"]
    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(
            project=f"plan_{cfg_dict['planner']['name']}", config=cfg_dict
        )
        wandb.run.name = "{}".format(output_dir.split("plan_outputs/")[-1])
    else:
        wandb_run = None

    ckpt_base_path = cfg_dict["ckpt_base_path"]
    model_name = cfg_dict.get("model_name")
    if ckpt_base_path.startswith("/"):
        model_path = ckpt_base_path
    else:
        model_path = f"{ckpt_base_path}/{cfg_dict['model_name']}/"
    model_path = os.path.abspath(model_path)
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)

    # Start the env worker processes before the model is on the GPU (pusht_human's workers
    # are spawned fresh processes, so each can open its own EGL context for the Panda renderer).
    # Only chunk_size workers: the n_evals episodes are planned and rolled out chunk_size at a
    # time (chunk_size=null -> all at once), so memory no longer grows with n_evals.
    n_envs = max(1, min(int(cfg_dict.get("chunk_size") or cfg_dict["n_evals"]), cfg_dict["n_evals"]))
    print(f"[plan.py] n_evals={cfg_dict['n_evals']} in chunks of {n_envs}", flush=True)
    # use dummy vector env for wall and deformable envs
    if model_cfg.env.name == "wall" or model_cfg.env.name == "deformable_env":
        from env.serial_vector_env import SerialVectorEnv
        env = SerialVectorEnv(
            [
                gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(n_envs)
            ]
        )
    else:
        env = SubprocVectorEnv(
            [
                lambda: gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(n_envs)
            ],
            # the Panda renderer needs its own EGL context per worker: fresh processes
            start_method="spawn" if model_cfg.env.name == "pusht_human" else None,
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed(cfg_dict["seed"])
    _, dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = dset["valid"]

    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = (
        Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    )
    model = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)
    t_after_model = time.perf_counter()
    print(f"[timing] setup_model_s={t_after_model - t_start:.3f}", flush=True)

    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset,
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
    )
    t_after_workspace = time.perf_counter()
    print(f"[timing] setup_workspace_s={t_after_workspace - t_after_model:.3f}", flush=True)

    logs = plan_workspace.perform_planning()
    t_after_planning = time.perf_counter()
    print(f"[timing] perform_planning_s={t_after_planning - t_after_workspace:.3f}", flush=True)
    print(f"[timing] total_planning_main_s={t_after_planning - t_start:.3f}", flush=True)
    return logs


@hydra.main(config_path="conf", config_name="plan_gd", version_base="1.1")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict["wandb_logging"] = bool(cfg_dict.get("wandb_logging", True))
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()
