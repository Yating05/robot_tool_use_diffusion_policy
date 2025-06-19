import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import zarr
from diffusion_policy.workspace.base_workspace import BaseWorkspace
if __name__ == "__main__":
    output_dir = "data/victor_eval_output"
    # if os.path.exists(output_dir):
    #     click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    

    # load checkpoint
    payload = torch.load(open("data/outputs/2025.06.19/12.52.40_train_diffusion_unet_hybrid_victor_diff/checkpoints/epoch=0250-train_action_mse_error=0.000.ckpt", 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    device = policy.device

    device = torch.device(device)
    policy.to(device)
    # policy.eval()
    zf = zarr.open("data/victor/victor_data.zarr", mode='r') #"data/pusht/pusht_cchi_v7_replay.zarr"

    # print(zf["data/image"][:10].shape)
    a = (np.moveaxis(np.array(zf["data/image"][:1]),-1,1)/255)[:, np.newaxis, :, :, :]
    # print(a.shape)
    # print(np.concatenate([a, a], axis=1).shape)

    # print(zf["data/robot_obs"][:10].shape)
    b = np.array(zf["data/robot_obs"][:1])[:, np.newaxis, :]
    obs = {
        "image" : np.concatenate([a, a], axis=1),
        # "image" : zf["data/image"][:20],
        "robot_obs" : np.concatenate([b, b], axis=1)
    }



    np_obs_dict = dict(obs)

    # print(np_obs_dict["image"].shape)
     # device transfer
    obs_dict = dict_apply(np_obs_dict, 
        lambda x: torch.from_numpy(x).to(
            device=device))

    # run policy
    with torch.no_grad():
        action_dict = policy.predict_action(obs_dict)

    # device_transfer
    np_action_dict = dict_apply(action_dict,
        lambda x: x.detach().to('cpu').numpy())

    action = np_action_dict['action']
    print(action.shape)
    print(action)