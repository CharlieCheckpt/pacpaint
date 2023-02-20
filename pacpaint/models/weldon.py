"""
Copyright (c) Owkin Inc.
This source code is licensed under the GNU GENERAL PUBLIC LICENSE v3 license found in the
LICENSE file in the root directory of this source tree.

Inspired from the paper Durand et al. 2016, WELDON: WEakly supervised 
Learning of Deep cOnvolutional neural Networks.
"""
import torch

from typing import Optional, List

from pacpaint.models.extreme_layer import ExtremeLayer
from pacpaint.models.mlp import TilesMLP


class Weldon(torch.nn.Module):
    """
    Weldon module.

    Parameters
    ----------
    in_features: int
    out_features: int
        controls the number of scores and, by extension, the number of out_features
    tiles_mlp_hidden: Optional[List[int]] = None
    bias: bool = True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        n_extreme: Optional[int] = 10,
        tiles_mlp_hidden: Optional[List[int]] = None,
        bias: bool = True,
    ):
        super(Weldon, self).__init__()

        self.score_model = TilesMLP(
            in_features, hidden=tiles_mlp_hidden, bias=bias, out_features=out_features
        )
        self.score_model.apply(self.weight_initialization)
        self.extreme_layer = ExtremeLayer(n_top=n_extreme, n_bottom=n_extreme)

    @staticmethod
    def weight_initialization(module: torch.nn.Module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, FEATURES)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        logits, extreme_scores: Tuple[torch.Tensor, torch.Tensor]:
            (B, OUT_FEATURES), (B, N_TOP + N_BOTTOM, OUT_FEATURES)
        """
        scores = self.score_model(x=x, mask=mask)
        extreme_scores = self.extreme_layer(x=scores, mask=mask)

        return torch.mean(extreme_scores, 1, keepdim=False)
