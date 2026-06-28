from torch.optim.lr_scheduler import LambdaLR
import numpy as np


class LambdaWarmUpCosineScheduler(LambdaLR):
    def __init__(self, optimizer, warm_up_steps, lr_min, lr_max, lr_start, max_decay_steps, verbosity_interval=0):
        self.warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.max_decay_steps = max_decay_steps
        self.verbosity_interval = verbosity_interval
        self.last_lr_multiplier = 0.
        # Pass self._schedule as the lambda — optimizer is handled by LambdaLR
        super().__init__(optimizer, lr_lambda=self._schedule)

    def _schedule(self, n):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f"current step: {n}, recent lr-multiplier: {self.last_lr_multiplier}")
        if n < self.warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.warm_up_steps * n + self.lr_start
        else:
            t = (n - self.warm_up_steps) / (self.max_decay_steps - self.warm_up_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(t * np.pi))
        self.last_lr_multiplier = lr
        return lr


class LambdaWarmUpCosineScheduler2(LambdaLR):
    """
    Supports repeated iterations, configurable via lists.
    """
    def __init__(self, optimizer, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):
        assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
        self.warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        self.cum_cycles = np.cumsum([0] + list(cycle_lengths))
        self.verbosity_interval = verbosity_interval
        self.last_f = 0.
        super().__init__(optimizer, lr_lambda=self._schedule)

    def _find_cycle(self, n):
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1

    def _schedule(self, n):
        cycle = self._find_cycle(n)
        n_in_cycle = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n_in_cycle % self.verbosity_interval == 0:
                print(f"current step: {n_in_cycle}, recent lr-multiplier: {self.last_f}, current cycle: {cycle}")
        if n_in_cycle < self.warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.warm_up_steps[cycle] * n_in_cycle + self.f_start[cycle]
        else:
            t = (n_in_cycle - self.warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.warm_up_steps[cycle])
            t = min(t, 1.0)
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (1 + np.cos(t * np.pi))
        self.last_f = f
        return f


class LambdaLinearScheduler(LambdaWarmUpCosineScheduler2):
    def _schedule(self, n):
        cycle = self._find_cycle(n)
        n_in_cycle = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n_in_cycle % self.verbosity_interval == 0:
                print(f"current step: {n_in_cycle}, recent lr-multiplier: {self.last_f}, current cycle: {cycle}")
        if n_in_cycle < self.warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.warm_up_steps[cycle] * n_in_cycle + self.f_start[cycle]
        else:
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (
                self.cycle_lengths[cycle] - n_in_cycle) / self.cycle_lengths[cycle]
        self.last_f = f
        return f