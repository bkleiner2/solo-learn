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
from solo.losses.barlow import barlow_loss_func
from solo.losses.simsiam import simsiam_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params


class BarlowTwinsMomentum(BaseMomentumMethod):
    def __init__(
        self, proj_hidden_dim: int, proj_output_dim: int, lamb: float, scale_loss: float, **kwargs
    ):
        """Implements Barlow Twins (https://arxiv.org/abs/2103.03230)

        Args:
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            proj_output_dim (int): number of dimensions of projected features.
            lamb (float): off-diagonal scaling factor for the cross-covariance matrix.
            scale_loss (float): scaling factor of the loss.
        """

        super().__init__(**kwargs)

        self.lamb = lamb
        self.scale_loss = scale_loss

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)


    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(BarlowTwinsMomentum, BarlowTwinsMomentum).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("barlow_twins")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=2048)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--lamb", type=float, default=0.0051)
        parser.add_argument("--scale_loss", type=float, default=0.024)
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to parent's learnable parameters.

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

    def forward(self, X, *args, **kwargs):
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
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
        
        # ------- loss calculations-------
        neg_cos_sim = 0
        cross_entropy = 0
        l1_dist = 0
        l2_dist = 0
        smooth_l1 = 0
        kl_div = 0
        barlow_loss = 0
        for v1 in range(self.num_large_crops):
            for v2 in np.delete(range(self.num_crops), v1):
                barlow_loss += barlow_loss_func(Z[v2], Z_momentum[v1].detach(), lamb=self.lamb, scale_loss=self.scale_loss) / len(feats)
                
                with torch.no_grad():
                    neg_cos_sim += simsiam_loss_func(Z[v2], Z_momentum[v1]) / len(feats)
                    l2_dist += F.mse_loss(Z[v2], Z_momentum[v1]) / len(feats)
                    l1_dist += F.l1_loss(Z[v2], Z_momentum[v1]) / len(feats)
                    cross_entropy += F.cross_entropy(Z[v2], Z_momentum[v1]) / len(feats)
                    smooth_l1 += F.smooth_l1_loss(Z[v2], Z_momentum[v1]) / len(feats)
                    kl_div += F.kl_div(Z[v2], Z_momentum[v1]) / len(feats)
        
        if self.training_labels is not None:
            self.training_labels[img_indexes] = torch.stack(Z_momentum).mean(dim=0).detach().cpu()

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
            "train_barlow_loss": barlow_loss,
            "train_mom_feats_std": mom_std,
            "train_z_std": z_std
        }
        

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return barlow_loss

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for Barlow Twins reusing BaseMomentumMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of Barlow loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        barlow_loss = self._shared_step(out["feats"], out["momentum_feats"], batch[0])

        return barlow_loss + class_loss
