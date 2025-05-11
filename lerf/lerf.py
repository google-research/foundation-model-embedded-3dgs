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

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

from torch import nn
import numpy as np
import torch
from torch.nn import Parameter

from lerf.encoders.openclip_encoder import OpenCLIPNetworkConfig
from lerf.lerf_field import LERFField
from lerf.lerf_fieldheadnames import LERFFieldHeadNames
# from lerf.lerf_renderers import CLIPRenderer, MeanRenderer
from scene.gaussian_model import GaussianModel


@dataclass
class LERFModelConfig:
    # _target: Type = field(default_factory=lambda: LERFModel)
    clip_loss_weight: float = 0.1
    n_scales: int = 30
    max_scale: float = 1.5
    """maximum scale used to compute relevancy with"""
    num_lerf_samples: int = 24
    hashgrid_layers: Tuple[int, ...] = (12, 12)
    hashgrid_resolutions: Tuple[Tuple[int, int], ...] = ((16, 128), (128, 512))
    hashgrid_sizes: Tuple[int, ...] = (19, 19)


class LERFModel(nn.Module):

    def __init__(self):
        # self.renderer_clip = CLIPRenderer()
        # self.renderer_mean = MeanRenderer()
        self.config = LERFModelConfig()
        super(LERFModel, self).__init__()
        self.image_encoder_featdim = OpenCLIPNetworkConfig().clip_n_dims # TODO: Rocky, check this
        self.lerf_field = LERFField(
            grid_layers=self.config.hashgrid_layers,
            grid_sizes=self.config.hashgrid_sizes,
            grid_resolutions=self.config.hashgrid_resolutions,
            clip_n_dims=self.image_encoder_featdim,
        )
        self.intermed_vlfeat_dim = self.lerf_field.intermed_vlfeat_dim

    def forward(self, gaussian_samples: GaussianModel, clip_scales, valid_gaussians_mask=None):
        outputs = {}
        lerf_field_outputs = self.lerf_field.get_outputs(gaussian_samples, clip_scales, valid_gaussians_mask) # The gaussian_samples.feature_vl is changed in this function
        return lerf_field_outputs

    def get_relevancy_img(self, fmap_embed, lerf_image_encoder, query_embed = None):
        '''
        fmap_embed: C, H, W
        '''
        H, W = fmap_embed.shape[1], fmap_embed.shape[2]
        fmap_embed = fmap_embed.view(fmap_embed.shape[0], -1).permute(1, 0).contiguous()
        n_phrases = len(lerf_image_encoder.positives)
        n_phrases_score = [None for _ in range(n_phrases)]

        for j in range(n_phrases):
            probs = lerf_image_encoder.get_relevancy(fmap_embed, j) # (HxW, 2)
            pos_prob = probs[..., 0:1]
            n_phrases_score[j] = pos_prob.view(H, W, 1)
        return torch.stack(n_phrases_score) # (n_positive_embeds, H, W, 1)


    def get_relevancy_img_segmentation(self, fmap_embed, lerf_image_encoder, query_embed = None):
        '''
        fmap_embed: C, H, W
        '''
        H, W = fmap_embed.shape[1], fmap_embed.shape[2]
        fmap_embed = fmap_embed.view(fmap_embed.shape[0], -1).permute(1, 0).contiguous() #(HxW, C)

        softmax = lerf_image_encoder.get_relevancy_segmentation(fmap_embed) # (HxW, nphrases)
        softmax = softmax.view(H, W, -1) # (H, W, nphrases)
        softmax = softmax.permute(2,0,1).contiguous() # (nphrases, H, W)
        return softmax.unsqueeze(-1) # (nphrases, H, W, 1)

    def get_max_across_rays(self, ray_samples, weights, hashgrid_field, scales_shape, preset_scales=None): # Note this is not used.
        # TODO smoothen this out
        if preset_scales is not None:
            assert len(preset_scales) == len(self.image_encoder.positives)
            scales_list = torch.tensor(preset_scales)
        else:
            scales_list = torch.linspace(0.0, self.config.max_scale, self.config.n_scales)

        # probably not a good idea bc it's prob going to be a lot of memory
        n_phrases = len(self.image_encoder.positives)
        n_phrases_maxs = [None for _ in range(n_phrases)]
        n_phrases_sims = [None for _ in range(n_phrases)]
        for i, scale in enumerate(scales_list):
            scale = scale.item()
            with torch.no_grad():
                clip_output = self.lerf_field.get_output_from_hashgrid(
                    ray_samples,
                    hashgrid_field,
                    torch.full(scales_shape, scale, device=weights.device, dtype=hashgrid_field.dtype),
                )
            clip_output = self.renderer_clip(embeds=clip_output, weights=weights.detach())

            for j in range(n_phrases):
                if preset_scales is None or j == i:
                    probs = self.image_encoder.get_relevancy(clip_output, j)
                    pos_prob = probs[..., 0:1]
                    if n_phrases_maxs[j] is None or pos_prob.max() > n_phrases_sims[j].max():
                        n_phrases_maxs[j] = scale
                        n_phrases_sims[j] = pos_prob
        return torch.stack(n_phrases_sims), torch.Tensor(n_phrases_maxs)
