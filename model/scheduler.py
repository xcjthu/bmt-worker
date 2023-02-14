
import math
from bmtrain.lr_scheduler.warmup import WarmupLRScheduler
import bmtrain as bmt


def get_scheduler(lrsche_type, optimizer, start_lr, warup_iter, end_iter, num_iter=0):
    type2class = {
        "linear": bmt.lr_scheduler.Linear,
        "t5": T5Scheduler,
        "constant": ConstantScheduler
    }
    lr_scheduler = type2class[lrsche_type](optimizer, start_lr=start_lr, warmup_iter=warup_iter, end_iter=end_iter, num_iter=num_iter)

    return lr_scheduler


class T5Scheduler(WarmupLRScheduler):
    r"""
        After a warmup period during which performs :math:`\text{lr}=\text{start_lr}\times \dfrac{\text{num_iter}}{\text{warmup_iter}^{3/2}}`,
        The decay period performs :math:`\text{lr}=\text{start_lr}\times \dfrac{\text{1}}{\sqrt{\text{num_iter}}}`
    """

    def get_lr_warmup(self, num_iter) -> float:
        return self.start_lr
    
    def get_lr_decay(self, num_iter) -> float:
        return self.start_lr * math.sqrt(self.warmup_iter) / math.sqrt(num_iter)

class ConstantScheduler(WarmupLRScheduler):
    r"""
        After a warmup period during which learning rate increases linearly between 0 and the start_lr,
        The decay period performs :math:`\text{lr}=\text{start_lr}`
    """
    def get_lr_warmup(self, num_iter) -> float:
        return self.start_lr
    
    def get_lr_decay(self, num_iter) -> float:
        return self.start_lr
