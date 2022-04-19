# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.simclr import simclr_loss_func
from solo.losses.simsiam import simsiam_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params


class SimCLRMomentum(BaseMomentumMethod):
    def __init__(self, proj_output_dim: int, proj_hidden_dim: int, temperature: float, **kwargs):
        """Implements SimCLR (https://arxiv.org/abs/2002.05709) with a momentum encoder.

        Args:
            proj_output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            temperature (float): temperature for the softmax in the contrastive loss.
        """

        super().__init__(**kwargs)

        self.temperature = temperature

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        
        # momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SimCLRMomentum, SimCLRMomentum).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simclr")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--temperature", type=float, default=0.1)

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs
    
    def forward(self, X: torch.tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the backbone, the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        return {**out, "z": z}
    
    def _shared_step(
        self, feats: List[torch.Tensor], momentum_feats: List[torch.Tensor], img_indexes: List[int]
    ) -> torch.Tensor:
        

        Z = [self.projector(f) for f in feats]

        # forward momentum backbone
        with torch.no_grad():
            Z_momentum = [self.momentum_projector(f) for f in momentum_feats]
        
        n_augs = self.num_large_crops + self.num_small_crops
        indexes = img_indexes.repeat(n_augs)
        
        # ------- loss -------
        neg_cos_sim = 0
        cross_entropy = 0
        l1_dist = 0
        l2_dist = 0
        smooth_l1 = 0
        kl_div = 0
        nce_loss = 0
        for v1 in range(self.num_large_crops):
            for v2 in np.delete(range(self.num_crops), v1):
                z = torch.cat([Z[v2], Z_momentum[v1].detach()])
                nce_loss += (simclr_loss_func(z, indexes=indexes, temperature=self.temperature) / n_augs)
                
                with torch.no_grad():
                    neg_cos_sim += simsiam_loss_func(Z[v2], Z_momentum[v1])
                    l2_dist += F.mse_loss(Z[v2], Z_momentum[v1])
                    l1_dist += F.l1_loss(Z[v2], Z_momentum[v1])
                    cross_entropy += F.cross_entropy(Z[v2], Z_momentum[v1])
                    smooth_l1 += F.smooth_l1_loss(Z[v2], Z_momentum[v1])
                    kl_div += F.kl_div(Z[v2], Z_momentum[v1])

        # calculate std of features
        with torch.no_grad():
            mom_std = F.normalize(torch.stack(momentum_feats), dim=-1).std(dim=1).mean()
            z_std = F.normalize(torch.stack(Z[: self.num_large_crops]), dim=-1).std(dim=1).mean()

        metrics = {
            "train_cross_entropy": cross_entropy,
            "train_l1_dist": l1_dist,
            "train_l2_dist": l2_dist,
            "train_smooth_l1": smooth_l1,
            "train_kl_div": kl_div,
            "train_neg_cos_sim": neg_cos_sim,
            "train_nce_loss": nce_loss,
            "train_mom_feats_std": mom_std,
            "train_z_std": z_std
        }

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return nce_loss

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """


        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        nce_loss = self._shared_step(out["feats"], out["momentum_feats"], batch[0])

        return nce_loss + class_loss
