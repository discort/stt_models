import os
import logging

import torch.distributed as dist


def setup_distributed(world_size,
                      rank,
                      backend="nccl",
                      init_method="env://"):
    """Perform env setup and initialization for distributed training"""
    if init_method == "env://":
        _set_env_vars(world_size, rank)
    if world_size > 1 and "OMP_NUM_THREADS" not in os.environ:
        logging.info("Setting OMP_NUM_THREADS == 1")
        os.environ["OMP_NUM_THREADS"] = "1"
    params = {
        "backend": backend,
        "init_method": init_method,
        "world_size": world_size,
        "rank": rank,
    }
    logging.info("Initializing distributed process group with %s", params)
    dist.init_process_group(**params)
    logging.info("Initialized distributed process group.")


def _set_env_vars(world_size, rank):
    for key, default in [("MASTER_ADDR", "localhost"), ("MASTER_PORT", "29500")]:
        if key not in os.environ:
            os.environ[key] = default

    #os.environ["WORLD_SIZE"] = str(world_size)
    #os.environ["RANK"] = str(rank)
    os.environ['GLOO_SOCKET_IFNAME'] = 'lo0'
