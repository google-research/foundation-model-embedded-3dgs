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

"""
Datamanager.
"""

from __future__ import annotations

import os.path as osp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from lerf.data.utils.dino_dataloader import DinoDataloader
from lerf.data.utils.pyramid_embedding_dataloader import PyramidEmbeddingDataloader
from lerf.encoders.image_encoder import BaseImageEncoder

from scene import Scene

class LERFFeatManager:  # pylint: disable=abstract-method
    """Basic stored clip/deno feature and the raw data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """
    def __init__(
        self,
        dataname: str,
        input_scene: Scene,
        image_encoder: BaseImageEncoder,
        device: Union[torch.device, str] = "cpu"
    ):

        self.image_encoder = image_encoder
        self.device = device

        scene_viewpoints = input_scene.getTrainCameras().copy() # This is a list of Camera
        images = [scene_viewpoints[i].original_image.to(self.device)[None, ...] for i in range(len(scene_viewpoints))] # original_image is with shape [C, H, W]
        self.length = len(images)
        images = torch.cat(images) # (B, C, H, W)

        cache_dir = f"outputs/{dataname}"
        clip_cache_path = Path(osp.join(cache_dir, f"clip_{self.image_encoder.name}"))
        dino_cache_path = Path(osp.join(cache_dir, f"dino_{DinoDataloader.dino_model_type}.npy"))
        # NOTE: cache config is sensitive to list vs. tuple, because it checks for dict equality
        self.dino_dataloader = DinoDataloader(
            image_list=images,
            device=self.device,
            cfg={"image_shape": list(images.shape[2:4])},
            cache_path=dino_cache_path,
        )
        torch.cuda.empty_cache()
        self.clip_interpolator = PyramidEmbeddingDataloader(
            image_list=images,
            device=self.device,
            cfg={
                "tile_size_range": [0.05, 0.5],
                "tile_size_res": 7,
                "stride_scaler": 0.5,
                "image_shape": list(images.shape[2:4]),
                "model_name": self.image_encoder.name,
            },
            cache_path=clip_cache_path,
            model=self.image_encoder,
        )
        torch.cuda.empty_cache()
        del images

    def __len__(self):
        return self.length

    def __call__(self, image_index):
        # image_index is a scalar
        deno_feat_map = self.dino_dataloader.img_call(image_index)
        clip_feat_map = self.clip_interpolator.img_call(image_index)
        return deno_feat_map, clip_feat_map

    #
    # def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
    #     """Returns the next batch of data from the train dataloader."""
    #     self.train_count += 1
    #     image_batch = next(self.iter_train_image_dataloader)
    #     assert self.train_pixel_sampler is not None
    #     batch = self.train_pixel_sampler.sample(image_batch)
    #     ray_indices = batch["indices"]
    #     ray_bundle = self.train_ray_generator(ray_indices)
    #     batch["clip"], clip_scale = self.clip_interpolator(ray_indices) # Random scale
    #     batch["dino"] = self.dino_dataloader(ray_indices)
    #     ray_bundle.metadata["clip_scales"] = clip_scale
    #     # assume all cameras have the same focal length and image width
    #     ray_bundle.metadata["fx"] = self.train_dataset.cameras[0].fx.item()
    #     ray_bundle.metadata["width"] = self.train_dataset.cameras[0].width.item()
    #     ray_bundle.metadata["fy"] = self.train_dataset.cameras[0].fy.item()
    #     ray_bundle.metadata["height"] = self.train_dataset.cameras[0].height.item()
    #     return ray_bundle, batch
