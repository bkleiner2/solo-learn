import math

import torch


@torch.no_grad()
def initialize_momentum_params(online_encoder, momentum_encoder):
    params_online = online_encoder.parameters()
    params_momentum = momentum_encoder.parameters()
    for po, pm in zip(params_online, params_momentum):
        pm.data.copy_(po.data)
        pm.requires_grad = False


class MomentumUpdater:
    def __init__(self, base_tau=0.996, final_tau=1.0):
        super().__init__()
        self.base_tau = base_tau
        self.cur_tau = base_tau
        self.final_tau = final_tau

    @torch.no_grad()
    def update(self, online_net, momentum_net):
        for op, mp in zip(online_net.parameters(), momentum_net.parameters()):
            mp.data = self.cur_tau * mp.data + (1 - self.cur_tau) * op.data

    def update_tau(self, cur_step, max_steps):
        self.cur_tau = (
            self.final_tau
            - (self.final_tau - self.base_tau) * (math.cos(math.pi * cur_step / max_steps) + 1) / 2
        )
