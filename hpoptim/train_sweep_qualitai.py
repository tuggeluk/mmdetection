"""Train Sweep QualitAI.

Sets up a training run with a Weights and Biases sweep.

Optimized parameters:
- model.bbox_head.r
- model.bbox_head.max_energy
- model.bbox_head.loss_cls.loss_weight
- model.bbox_head.loss_bbox.loss_weight
- model.bbox_head.loss_energy.loss_weight
- model.bbox_head.loss_energy.gamma
- model.bbox_head.loss_energy.alpha
- lr_config.warmup
- lr_config.warump_iters
- lr_config.warmup_ratio
- optimizer.lr

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    January 23, 2020
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime

import torch


def parse_arguments():
    """Parses all the arguments for the sweep."""
    desc = "you shouldn't be reading this as this should never be run manually."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--max_energy', type=int)
    parser.add_argument('--loss_bbox_weight', type=float)
    parser.add_argument('--loss_energy_weight', type=float)
    parser.add_argument('--loss_energy_gamma', type=float)
    parser.add_argument('--loss_energy_alpha', type=float)
    parser.add_argument('--lr_warmup', type=str)
    parser.add_argument('--lr_warmup_iters', type=int)
    parser.add_argument('--lr_warmup_ratio', type=float)
    parser.add_argument('--lr', type=float)

    return parser.parse_args()


def create_run_config(base_config, name,
                      max_energy,
                      loss_bbox_weight,
                      loss_energy_weight,
                      loss_energy_gamma,
                      loss_energy_alpha,
                      lr_warmup,
                      lr_warmup_iters,
                      lr_warmup_ratio,
                      lr
                      ):
    """Creates a run configuration based off a base config.

    Returns:
        str: The path of the new configuration file.
    """
    with open(base_config, 'r') as file:
        config = file.readlines()

    config[41] = '        max_energy={},\n'.format(max_energy)

    config[53] = '            loss_weight={}),\n'.format(loss_bbox_weight)
    config[59] = '            loss_weight={},\n'.format(loss_energy_weight)

    config[57] = '            gamma={},\n'.format(loss_energy_gamma)
    config[58] = '            alpha={},\n'.format(loss_energy_alpha)

    config[138] = '    lr={},\n'.format(lr)
    config[154] = '    warmup="{}",\n'.format(lr_warmup)
    config[155] = '    warmup_iters={},\n'.format(lr_warmup_iters)
    config[156] = '    warmup_ratio={},\n'.format(lr_warmup_ratio)

    out_path = "/workspace/mmdetection/work_dirs/wfcos_qualitai_sweep_config/"
    out_path += name
    with open(out_path, 'w') as file:
        file.writelines(config)

    return out_path


def main(arguments):

    # world size in terms of number of processes/GPUs
    dist_world_size = torch.cuda.device_count()

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = '127.0.0.1'
    current_env["MASTER_PORT"] = str(29500)
    current_env["WORLD_SIZE"] = str(dist_world_size)

    processes = []

    if 'OMP_NUM_THREADS' not in os.environ:
        current_env["OMP_NUM_THREADS"] = str(12)
        print("*****************************************\n"
              "Setting OMP_NUM_THREADS environment variable for each process "
              "to be {} in default, to avoid your system being overloaded, "
              "please further tune the variable for optimal performance in "
              "your application as needed. \n"
              "*****************************************".format(
            current_env["OMP_NUM_THREADS"]))

    for local_rank in range(0, dist_world_size):
        # each process's rank
        dist_rank = local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        cmd = [sys.executable, "-u"]

        cmd.append("/workspace/mmdetection/tools/train.py")

        cmd.append("--local_rank={}".format(local_rank))

        config_name = "sweep--"\
                      + datetime.now().strftime("%Y_%m_%d--%H_%M_%S.py")
        config_path = create_run_config(
            "/workspace/mmdetection/configs/wfcos/wfcos_qualitai_sweep.py",
            config_name,
            **vars(arguments)
        )

        cmd.extend([config_path, "--validate", '--launcher', 'pytorch'])

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            returncode = process.returncode
            for p in processes:
                try:
                    p.terminate()
                except:
                    pass
            raise subprocess.CalledProcessError(
                returncode=returncode,
                cmd=cmd)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
