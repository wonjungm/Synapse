import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from tspipe.communicator import CommunicatorParam
from tspipe.gpu_worker import GpuWorker


def dummy_loss_fn(output, target):
    return output.sum()


def dummy_update_target_fn(*args, **kwargs):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--num_ubatch", type=int, default=None)
    parser.add_argument("--num_bwd_ubatch", type=int, default=None)
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=str, default="29500")
    args = parser.parse_args()

    torch.cuda.set_device(args.rank % torch.cuda.device_count())

    communicator_param = CommunicatorParam(
        rank=args.rank,
        world_size=args.world_size,
        master_addr=args.master_addr,
        master_port=args.master_port,
        num_partition=args.world_size,
        scheduler_process_rank=0,
    )

    # dummy optimizer (실제 optimizer는 init_ctx에서 overwrite됨)
    dummy_param = nn.Parameter(torch.zeros(1, device="cuda"))
    optimizer = optim.SGD([dummy_param], lr=0.1)

    worker = GpuWorker(
        partition_id=args.rank,
        num_ubatch=args.num_ubatch,
        num_bwd_ubatch=args.num_bwd_ubatch,
        communicator_param=communicator_param,
        optimizer=optimizer,
        momentum=0.9,
        loss_fn=dummy_loss_fn,
        update_target_fn=dummy_update_target_fn,
    )

    # 🔴 핵심: 이 프로세스 자체가 GPU worker
    worker.process.join()


if __name__ == "__main__":
    main()
