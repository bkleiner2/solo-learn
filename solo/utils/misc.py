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

import math
import warnings
from typing import List, Tuple
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn


def _1d_filter(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.isfinite()


def _2d_filter(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.isfinite().all(dim=1)


def _single_input_filter(tensor: torch.Tensor) -> Tuple[torch.Tensor]:
    if len(tensor.size()) == 1:
        filter_func = _1d_filter
    elif len(tensor.size()) == 2:
        filter_func = _2d_filter
    else:
        raise RuntimeError("Only 1d and 2d tensors are supported.")

    selected = filter_func(tensor)
    tensor = tensor[selected]

    return tensor, selected


def _multi_input_filter(tensors: List[torch.Tensor]) -> Tuple[torch.Tensor]:
    if len(tensors[0].size()) == 1:
        filter_func = _1d_filter
    elif len(tensors[0].size()) == 2:
        filter_func = _2d_filter
    else:
        raise RuntimeError("Only 1d and 2d tensors are supported.")

    selected = filter_func(tensors[0])
    for tensor in tensors[1:]:
        selected = torch.logical_and(selected, filter_func(tensor))
    tensors = [tensor[selected] for tensor in tensors]

    return tensors, selected


def filter_inf_n_nan(tensors: List[torch.Tensor], return_indexes: bool = False):
    """Filters out inf and nans from any tensor.
    This is usefull when there are instability issues,
    which cause a small number of values to go bad.

    Args:
        tensor (List): tensor to remove nans and infs from.

    Returns:
        torch.Tensor: filtered view of the tensor without nans or infs.
    """

    if isinstance(tensors, torch.Tensor):
        tensors, selected = _single_input_filter(tensors)
    else:
        tensors, selected = _multi_input_filter(tensors)

    if return_indexes:
        return tensors, selected
    return tensors


class FilterInfNNan(nn.Module):
    def __init__(self, module):
        """Layer that filters out inf and nans from any tensor.
        This is usefull when there are instability issues,
        which cause a small number of values to go bad.

        Args:
            tensor (List): tensor to remove nans and infs from.

        Returns:
            torch.Tensor: filtered view of the tensor without nans or infs.
        """
        super().__init__()

        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.module(x)
        out = filter_inf_n_nan(out)
        return out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "module":
                raise AttributeError()
            return getattr(self.module, name)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Copy & paste from PyTorch official master until it's in a few official releases - RW
    Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    def norm_cdf(x):
        """Computes standard normal cumulative distribution function"""

        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Copy & paste from PyTorch official master until it's in a few official releases - RW
    Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


class GatherLayer(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if dist.is_available() and dist.is_initialized():
            all_gradients = torch.stack(grads)
            dist.all_reduce(all_gradients)
            grad_out = all_gradients[get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)


class TrainingDataLog:
    def __init__(self):
        self.data_log = {}
        self.counts = defaultdict(int)

    def update(self, key, value):
        self.counts[key] += 1
        if self.counts[key] == 1:
            self.data_log[key] = value
        else:
            count = self.counts[key]
            self.data_log[key] = (self.data_log[key] * (count - 1) + value) / count




