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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from solo.losses.byol import byol_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params


class BYOL(BaseMomentumMethod):
    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
        **kwargs,
    ):
        """Implements BYOL (https://arxiv.org/abs/2006.07733).

        Args:
            proj_output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
        """

        super().__init__(**kwargs)

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )

        self.loss_function_to_use = kwargs.get("loss_function_to_use", "neg_cos_sim")

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(BYOL, BYOL).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("byol")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=256)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # predictor
        parser.add_argument("--pred_hidden_dim", type=int, default=512)

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        return {**out, "z": z, "p": p}

    def _shared_step(
        self, feats: List[torch.Tensor], momentum_feats: List[torch.Tensor], img_indexes: List[int]
    ) -> torch.Tensor:

        Z = [self.projector(f) for f in feats]
        P = [self.predictor(z) for z in Z]

        # forward momentum backbone
        with torch.no_grad():
            Z_momentum = [self.momentum_projector(f) for f in momentum_feats]
        
        # ------- negative consine similarity loss -------
        neg_cos_sim = 0
        cross_entropy = 0
        l1_dist = 0
        l2_dist = 0
        smooth_l1 = 0
        kl_div = 0
        for v1 in range(self.num_large_crops):
            for v2 in np.delete(range(self.num_crops), v1):
                p_detach = P[v2].detach()
                z_detach = Z_momentum[v1].detach()
                #TODO generalize this to all options
                if self.loss_function_to_use == "neg_cos_sim":
                    neg_cos_sim += byol_loss_func(P[v2], Z_momentum[v1])
                    l2_dist += F.mse_loss(p_detach, z_detach)
                elif self.loss_function_to_use == "l2_dist":
                    neg_cos_sim += byol_loss_func(p_detach, z_detach)
                    l2_dist += F.mse_loss(P[v2], z_detach)
                else:
                    raise ValueError("Only neg_cos_sim and l2_dist are supported")
                cross_entropy += F.cross_entropy(p_detach, z_detach)
                l1_dist += F.l1_loss(p_detach, z_detach)
                smooth_l1 += F.smooth_l1_loss(p_detach, z_detach)
                kl_div += F.kl_div(p_detach, z_detach)

        metrics = {
            "train_feats_cross_entropy": cross_entropy,
            "train_feats_l1_dist": l1_dist,
            "train_feats_l2_dist": l2_dist,
            "train_feats_smooth_l1": smooth_l1,
            "train_feats_kl_div": kl_div,
            "train_neg_cos_sim": neg_cos_sim,
        }
        
        if self.training_labels is not None:
            self.training_labels[img_indexes] = torch.stack(Z_momentum).mean(dim=0).detach().cpu()

        # calculate std of features
        with torch.no_grad():
            mom_std = F.normalize(torch.stack(momentum_feats), dim=-1).std(dim=1).mean()
            z_std = F.normalize(torch.stack(Z[: self.num_large_crops]), dim=-1).std(dim=1).mean()

        metrics["train_mom_feats_std"] = mom_std
        metrics["train_z_std"] = z_std

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        # keeping this consistent with prior runs so we can compare easily
        key_prefix = "train_" if self.loss_function_to_use == "neg_cos_sim" else "train_feats_"

        return metrics[key_prefix + self.loss_function_to_use], z_std

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for BYOL reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of BYOL and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        byol_loss, z_std = self._shared_step(out["feats"], out["momentum_feats"], batch[0])

        return byol_loss + class_loss
