import math
import torch
from importlib import import_module


# -- cos + warmup
def cos_func(warmup_epochs, total_epochs, min_ratio):
    def inner_func(x):
        if x <= warmup_epochs:
            return min_ratio + (1 - min_ratio) * (x / warmup_epochs)
        else:
            return min_ratio + (1 - min_ratio) \
                   * (1 + math.cos((x - warmup_epochs) * math.pi / (total_epochs - warmup_epochs))) / 2
    return inner_func


def get_cosine_scheduler(opt: torch.optim.Optimizer,
                         warmup_ratio: float,
                         total_epochs: int,
                         min_ratio: float):
    """
    linear warmup 이후 cosine lr decay를 적용하는 scheduler를 반환

    Args:
        opt:
            torch optimizer
        warmup_ratio:
            int(total_epochs * warmup_ratio) 만큼의 epoch동안 warmup을 진행한다.
        total_epochs:
            전체 학습 epochs 수
        min_ratio:
            lr의 시작값 = 종료값 = min_ratio * optimizer.lr
    """
    warmup_epochs = int(warmup_ratio * total_epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=cos_func(warmup_epochs, total_epochs, min_ratio)
    )
    return scheduler


schedulers = {
    'warmup_cosine': get_cosine_scheduler
}


def get_scheduler_module(scheduler_name):
    if scheduler_name in schedulers:
        return schedulers[scheduler_name]
    else:
        from importlib import import_module
        return getattr(import_module("torch.optim.lr_scheduler"), scheduler_name)
