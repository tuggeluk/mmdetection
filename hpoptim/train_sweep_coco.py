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
from signal import SIGKILL, SIGTERM
import subprocess
import sys
from datetime import datetime
import socket
from time import sleep

import torch


def parse_arguments():
    """Parses all the arguments for the sweep."""
    desc = "you shouldn't be reading this as this should never be run manually."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--max_energy', type=int)
    parser.add_argument('--loss_class_weight', type=float)
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
                      loss_class_weight,
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

    config[41] = f'        max_energy={max_energy},\n'

    config[50] = f'            loss_weight={loss_class_weight}),\n'
    config[53] = f'            loss_weight={loss_bbox_weight}),\n'
    config[59] = f'            loss_weight={loss_energy_weight},\n'

    config[57] = f'            gamma={loss_energy_gamma},\n'
    config[58] = f'            alpha={loss_energy_alpha},\n'

    config[138] = f'    lr={lr},\n'
    config[149] = f'    warmup="{lr_warmup}",\n'
    config[150] = f'    warmup_iters={lr_warmup_iters},\n'
    config[151] = f'    warmup_ratio={lr_warmup_ratio},\n'

    out_dir = "/workspace/mmdetection/work_dirs/wfcos_coco_sweep_config/"
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir)
        except OSError:
            pass
    out_path = out_dir + name
    with open(out_path, 'w') as file:
        file.writelines(config)

    return out_path


def test_prev_pids():
    """Makes sure previous process are dead.

    Make sure prev processes have been terminated. Otherwise, wait up to 5
    minutes to let them finish up whatever it is they're doing. If processes
    still haven't died, then kill them.
    """
    retries = 0
    prev_processes = []
    # First get all prev processes
    try:
        with open('/tmp/mmdet_pids.txt', 'r') as pid_file:
            for line in pid_file:
                prev_processes.append(line.strip())
    except FileNotFoundError:
        pass

    while retries <= 60 and prev_processes:
        retries += 1
        proc_list = (subprocess.Popen(["ps", "-o", "pid"],
                                      stdout=subprocess.PIPE)
                     .communicate())[0].decode('utf-8').splitlines()[1:]
        for pid in prev_processes:
            # Check to see if there are any processes still running. If they
            # have already been killed of have died, remove them from the list.
            if pid not in proc_list:
                prev_processes.remove(pid)

        if prev_processes:
            for pid in prev_processes:
                try:
                    os.kill(int(pid), SIGTERM)
                except ProcessLookupError:
                    if os.path.exists('/tmp/mmdet_pids.txt'):
                        os.remove('/tmp/mmdet_pids.txt')
            sleep(5.)

    # Final sigkill if there are still processes
    if prev_processes:
        for pid in prev_processes:
            os.kill(int(pid), SIGKILL)


def get_available_port(addr):
    """Tries to get the next available port.

    Args:
        addr (str): Address to test ports on.

    Raises:
        ConnectionError: If no ports are available a ConnectionError will be
            raised.
    Returns:
        int: The next available port.
    """
    port = 32768

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        while s.connect_ex((addr, port)) == 0 and port < 61001:
            port += 1

    if port >= 61001:
        raise ConnectionError('No ports are available for torch.distributed')

    return port


def main(arguments):

    # world size in terms of number of processes/GPUs
    dist_world_size = torch.cuda.device_count()

    test_prev_pids()

    # Now final check to make sure that address and port is valid and available.
    addr = '127.0.0.1'
    port = get_available_port(addr)

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = addr
    current_env["MASTER_PORT"] = str(port)
    current_env["WORLD_SIZE"] = str(dist_world_size)

    processes = []
    process_pids = []

    if 'OMP_NUM_THREADS' not in os.environ:
        current_env["OMP_NUM_THREADS"] = str(12)
        print("*****************************************\n"
              "Setting OMP_NUM_THREADS environment variable for each process "
              "to be {} in default, to avoid your system being overloaded, "
              "please further tune the variable for optimal performance in "
              "your application as needed. \n"
              "*****************************************"
              .format(current_env["OMP_NUM_THREADS"]))

    for local_rank in range(0, dist_world_size):
        # each process's rank
        dist_rank = local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        cmd = [
            sys.executable,
            "-u",
            "/workspace/mmdetection/tools/train.py",
            "--local_rank={}".format(local_rank)
        ]

        config_name = "sweep--"\
                      + datetime.now().strftime("%Y_%m_%d--%H_%M_%S.py")
        config_path = create_run_config(
            "/workspace/mmdetection/configs/wfcos/wfcos_coco_sweep.py",
            config_name,
            **vars(arguments)
        )

        cmd.extend([config_path, "--validate", '--launcher', 'pytorch'])

        process = subprocess.Popen(cmd, env=current_env)

        processes.append(process)
        process_pids.append('{}\n'.format(process.pid))

    # Write process PIDs so we can check for them next run.
    with open('/tmp/mmdet_pids.txt', 'w') as pid_file:
        pid_file.writelines(process_pids)

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
