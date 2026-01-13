import logging
import os

from time import perf_counter

import torch
import torch.distributed as dist

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Env var are preset when launching the benchmark
    env_rank = os.environ.get("RANK", 0)
    env_world_size = os.environ.get("WORLD_SIZE", 1)
    env_master_addr = os.environ.get("MASTER_ADDR", "localhost")
    env_master_port = os.environ.get("MASTER_PORT", "23456")

    start = perf_counter()
    tcp_store = dist.TCPStore(
        env_master_addr,
        int(env_master_port),
        world_size=int(env_world_size),
        is_master=(int(env_rank) == 0),
    )
    end = perf_counter()
    time_elapsed = end - start
    logger.info(
        f"Complete TCPStore init with rank={env_rank}, world_size={env_world_size} in {time_elapsed} seconds."
    )
