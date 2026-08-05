import copy
import torch
import lightning as L


class EMACallback(L.Callback):
    def __init__(self, decay=0.999):
        super().__init__()
        self._decay = decay
        self._ema_state = None
        self._backup = None

    def on_fit_start(self, trainer, pl_module):
        self._ema_state = copy.deepcopy(pl_module.state_dict()){

    @torch.no_grad()
    def on_train_batch_ends(self, trainer, pl_module):
        for name, param in pl_module.state_dict().items():
            if param.dtype.is_floating_point:
                self._ema_state[name].mul_(self._decay).add_(
                    param.detach(), alpha=1.0 - self._decay
                )
            else:
                self._ema_state[name].copy_(param)  # non-float buffers like counters

    def on_train_end(self, trainer, pl_module):
        # swap the weights back in
        self._backup = copy.deepcopy(pl_module.state_dict())
        pl_module.load_state_dict(self._ema_state)
