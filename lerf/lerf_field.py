# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from lerf.lerf_fieldheadnames import LERFFieldHeadNames
from torch import nn, Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import sys
from jaxtyping import Float

from scene.gaussian_model import GaussianModel
from myutils.general_utils import recover_symetric

try:
    import tinycudann as tcnn
except ImportError:
    pass
except EnvironmentError as _exp:
    if "Unknown compute capability" not in _exp.args[0]:
        raise _exp
    print("Could not load tinycudann: " + str(_exp), file=sys.stderr)


# Refer to nerfstudio.spatial_distortion, https://docs.nerf.studio/nerfology/model_components/visualize_spatial_distortions.html
class SceneContraction(nn.Module):
    """Contract unbounded space using the contraction was proposed in MipNeRF-360.
        We use the following contraction equation:

        .. math::

            f(x) = \\begin{cases}
                x & ||x|| \\leq 1 \\\\
                (2 - \\frac{1}{||x||})(\\frac{x}{||x||}) & ||x|| > 1
            \\end{cases}

        If the order is not specified, we use the Frobenius norm, this will contract the space to a sphere of
        radius 2. If the order is L_inf (order=float("inf")), we will contract the space to a cube of side length 4.
        If using voxel based encodings such as the Hash encoder, we recommend using the L_inf norm.

        Args:
            order: Order of the norm. Default to the Frobenius norm. Must be set to None for Gaussians.

    """

    def __init__(self, order: Optional[Union[float, int]] = None) -> None:
        super().__init__()
        self.order = order

    def forward(self, positions):
        def contract(x):
            mag = torch.linalg.norm(x, ord=self.order, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

        return contract(positions)



class LERFField(nn.Module):
    def __init__(
        self,
        grid_layers: Tuple[int, ...],
        grid_sizes: Tuple[int, ...],
        grid_resolutions: Tuple[Tuple[int, int], ...],
        clip_n_dims: int,
    ):
        super(LERFField, self).__init__()
        self.spatial_distortion = SceneContraction()
        assert len(grid_layers) == len(grid_sizes) and len(grid_resolutions) == len(grid_layers)
        self.clip_encs = torch.nn.ModuleList(
            [
                LERFField._get_encoding(
                    grid_resolutions[i][0], grid_resolutions[i][1], grid_layers[i], indim=3, hash_size=grid_sizes[i]
                )
                for i in range(len(grid_layers))
            ]
        )
        self.intermed_vlfeat_dimlist = [e.n_output_dims for e in self.clip_encs]
        tot_out_dims = sum(self.intermed_vlfeat_dimlist)
        self.intermed_vlfeat_dim = tot_out_dims

        self.clip_net = tcnn.Network(
            n_input_dims=tot_out_dims, # + 1,
            n_output_dims=clip_n_dims,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 4,
            },
        )

        self.dino_net = tcnn.Network(
            n_input_dims=tot_out_dims,
            n_output_dims=384, # 384 for dino, 64 for pca compressed dino, 768 for dinov2?
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 1,
            },
        )

    @staticmethod
    def _get_encoding(start_res, end_res, levels, indim=3, hash_size=19):
        growth = np.exp((np.log(end_res) - np.log(start_res)) / (levels - 1))
        enc = tcnn.Encoding(
            n_input_dims=indim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": levels,
                "n_features_per_level": 8,
                "log2_hashmap_size": hash_size,
                "base_resolution": start_res,
                "per_level_scale": growth,
            },
        )
        return enc

    def get_outputs(self, gaussian_samples: GaussianModel, clip_scales, valid_gaussians_mask=None) -> Dict[LERFFieldHeadNames, Float[Tensor, "bs dim"]]:
        # random scales, one scale
        outputs = {}
        positions = (gaussian_samples.get_xyz[valid_gaussians_mask]).clone()
        # Refer to nerfstudio.spatial_distortion https://docs.nerf.studio/nerfology/model_components/visualize_spatial_distortions.html
        # Note that only the mean of the gaussians are distorted here, while the covariance is exempted.
        positions = self.spatial_distortion(positions) # Normalize the positions into [-2, 2]
        positions = (positions + 2.0) / 4.0  # Normalize the positions into [-1, 1]

        xs = [e(positions.view(-1, 3)) for e in self.clip_encs]
        x = torch.concat(xs, dim=-1)

        clip_pass = self.clip_net(x).view(positions.shape[0], -1)
        clip_pass = clip_pass + 1e-8 # This can be enabled to stabelize the training.
        clip_pass = F.normalize(clip_pass, dim=-1, eps=1e-4) # e31cca45-5, this also incurs nan gradient
        outputs[LERFFieldHeadNames.CLIP] = clip_pass.float()

        dino_pass = self.dino_net(x).view(positions.shape[0], -1)
        dino_pass = dino_pass.float()
        outputs[LERFFieldHeadNames.DINO] = dino_pass
        if clip_pass.isnan().any() or dino_pass.isnan().any():
            print(f"---- The ratio of Nan: {dino_pass.sum(dim=1).isnan().sum()}/{dino_pass.shape[0]}, dino_pass: {dino_pass}")
            print(f"---- The ratio of Nan: {clip_pass.sum(dim=1).isnan().sum()}/{clip_pass.shape[0]}, clip_pass: {clip_pass}")
        return outputs

