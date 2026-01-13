import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp



def init_process(rank, world_size):
    print(f"进程已启动: 此进程的 rank 是 {rank}")


def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    world_size = torch.cuda.device_count()
    print(f"准备启动 {world_size} 个进程...")
    mp.spawn(
        init_process,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
