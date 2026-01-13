<details>
<summary>init_process_group</summary>

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_process(rank, world_size):
    print(f"进程已启动: 此进程的 rank 是 {rank}")
    
    # 设置当前进程使用的 GPU
    torch.cuda.set_device(rank)
    
    try:
        # 加入进程组
        print(f"进程 {rank} 正在加入进程组...")
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        print(f"进程 {rank} 已成功加入进程组")
        
        # 验证身份
        assert rank == dist.get_rank()
        assert world_size == dist.get_world_size()
        
        # 准备当前进程的信息
        process_info = (
            f"\n进程 {rank} 信息:\n"
            f"- Device: {torch.cuda.current_device()}\n"
            f"- GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}\n"
        )
        
        # 将字符串转换为固定长度的张量
        max_len = 100  # 确保足够长以容纳信息
        process_info_tensor = torch.zeros(max_len, dtype=torch.int32, device='cuda')
        process_info_bytes = process_info.encode('utf-8')
        process_info_tensor[:len(process_info_bytes)] = torch.tensor([b for b in process_info_bytes], dtype=torch.int32)
        
        # 创建用于收集所有进程信息的张量列表
        gathered_tensors = [torch.zeros(max_len, dtype=torch.int32, device='cuda') for _ in range(world_size)]

        # 使用 all_gather 收集所有进程的信息
        dist.all_gather(gathered_tensors, process_info_tensor)


        if rank == 0:
            print("=============== 所有进程信息 ===============")
            for tensor in gathered_tensors:
                info_bytes = tensor.cpu().numpy().astype('uint8').tobytes() 
                info_str = info_bytes.decode('utf-8', 'ignore').strip('\x00')
                print(info_str)
        
        # 创建张量并进行通信
        tensor = torch.ones(1).cuda() * rank
        print(f"进程 {rank} 的原始张量值: {tensor.item()}")
        
        # 所有进程同步点
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"进程 {rank} 的最终张量值: {tensor.item()}")
    
    finally:
        dist.destroy_process_group()

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
    
    #! 等价于通过以下代码启动进程
    # processes = []
    # for rank in range(world_size):
    #     p = mp.Process(target=init_process, args=(rank, world_size))
    #     p.start()
    #     processes.append(p)

    # # 相当于 join=True 的效果
    # for p in processes:
    #     p.join()

if __name__ == "__main__":
    main()
```

</details> 