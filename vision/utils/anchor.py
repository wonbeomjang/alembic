import copy
from typing import List, Dict

import torch
from torch import nn, Tensor


class AnchorGenerator(nn.Module):
    """
    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map. This module assumes aspect ratio = height / width for
    each anchor.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Args:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """

    __annotations__ = {
        "cell_anchors": List[torch.Tensor],
    }

    def __init__(
        self,
        size=(2 ** (0.0 / 3), 2 ** (1.0 / 3), 2 ** (2.0 / 3)),
        aspect_ratio=(0.5, 1.0, 2.0),
    ):
        super().__init__()

        self.size = size
        self.aspect_ratio = aspect_ratio
        self.cell_anchors = self.generate_anchors(size, aspect_ratio)

    def generate_anchors(
        self,
        scales: List[int],
        aspect_ratios: List[float],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors

    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device):
        self.cell_anchors = torch.stack(
            [
                cell_anchor.to(dtype=dtype, device=device)
                for cell_anchor in self.cell_anchors
            ]
        )

    def num_anchors_per_location(self) -> List[int]:
        return [len(s) * len(a) for s, a in zip(self.size, self.aspect_ratio)]

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(
        self, grid_sizes: List[List[int]], strides: List[List[Tensor]]
    ) -> List[Tensor]:
        anchors = []
        torch._assert(self.cell_anchors is not None, "cell_anchors should not be None")
        torch._assert(
            len(grid_sizes) == len(strides),
            "Anchors should be Tuple[Tuple[int]] because each feature "
            "map could potentially have different sizes and aspect ratios. "
            "There needs to be a match between the number of "
            "feature maps passed and the number of sizes / aspect ratios specified.",
        )

        for size, stride in zip(grid_sizes, strides):
            cell_anchors = copy.deepcopy(self.cell_anchors)

            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = cell_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = (
                torch.arange(0, grid_width, dtype=torch.int32, device=device)
                * stride_width
            )
            shifts_y = (
                torch.arange(0, grid_height, dtype=torch.int32, device=device)
                * stride_height
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            cell_anchors[..., 0] *= grid_width
            cell_anchors[..., 1] *= grid_height
            cell_anchors[..., 2] *= grid_width
            cell_anchors[..., 3] *= grid_height
            anchors.append(
                (shifts.view(-1, 1, 4) + cell_anchors.view(1, -1, 4))
                .reshape(-1, 4)
                .round()
            )
        return anchors

    def forward(
        self, image_list: List[Tensor], feature_maps: Dict[str, Tensor]
    ) -> List[Tensor]:
        min_key = min(feature_maps.keys())
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps.values()]
        image_size = image_list[0].shape[-2:]

        dtype, device = feature_maps[min_key].dtype, feature_maps[min_key].device
        strides = [
            [
                torch.empty((), dtype=torch.int64, device=device).fill_(
                    image_size[0] // g[0]
                ),
                torch.empty((), dtype=torch.int64, device=device).fill_(
                    image_size[1] // g[1]
                ),
            ]
            for g in grid_sizes
        ]
        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        anchors: List[List[torch.Tensor]] = []
        for _ in range(len(image_list)):
            anchors_in_image = [
                anchors_per_feature_map
                for anchors_per_feature_map in anchors_over_all_feature_maps
            ]
            anchors.append(anchors_in_image)
        result = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        return result
