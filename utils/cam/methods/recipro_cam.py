import numpy as np

import torch
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor


class ReciproCam:
    def __init__(self, model: torch.nn.Module):
        device = next(model.parameters()).device
        self.feature_net = create_feature_extractor(model, return_nodes={"layer4": "layer"}).eval()
        self.head_net = nn.Sequential(model.avgpool, nn.Flatten(), model.fc.eval())
        self.device = device
        self.feature = None
        self.softmax = torch.nn.Softmax(dim=1)
        self.gaussian = torch.tensor(
            [
                [1 / 16.0, 1 / 8.0, 1 / 16.0],
                [1 / 8.0, 1 / 4.0, 1 / 8.0],
                [1 / 16.0, 1 / 8.0, 1 / 16.0],
            ]
        ).to(device)

    def _mosaic_feature(self, feature_map, is_gaussian=False):
        _, num_channel, height, width = feature_map.shape
        new_features = torch.zeros(height * width, num_channel, height, width).to(
            self.device
        )
        if is_gaussian is False:
            for k in range(height * width):
                for i in range(height):
                    for j in range(width):
                        if k == i * width + j:
                            new_features[k, :, i, j] = feature_map[0, :, i, j]
        else:
            for k in range(height * width):
                for i in range(height):
                    kx_s, kx_e = max(i - 1, 0), min(i + 1, height - 1)
                    sx_s = 1 if i == 0 else 0
                    sx_e = 1 if i == height - 1 else 2
                    for j in range(width):
                        ky_s, ky_e = max(j - 1, 0), min(j + 1, width - 1)
                        sy_s = 1 if j == 0 else 0
                        sy_e = 1 if j == width - 1 else 2
                        if k == i * width + j:
                            r_feature_map = (
                                feature_map[0, :, i, j]
                                .reshape(num_channel, 1, 1)
                                .repeat(1, self.gaussian.shape[0], self.gaussian.shape[1])
                            )
                            score_map = r_feature_map * self.gaussian.repeat(
                                num_channel, 1, 1
                            )
                            new_features[
                                k, :, kx_s : kx_e + 1, ky_s : ky_e + 1
                            ] = score_map[:, sx_s : sx_e + 1, sy_s : sy_e + 1]

        return new_features

    def _weight_accum(self, mosaic_predict, class_index, height, width):
        cam = torch.zeros(height, width).to(self.device)
        for i in range(height):
            for j in range(width):
                cam[i, j] = mosaic_predict[i * width + j][class_index]
        return cam

    def __call__(self, image, class_index=None):
        feature_map = self.feature_net(image)['layer']

        prediction = self.head_net(feature_map)

        if class_index is None:
            class_index = np.argmax(prediction.cpu().data.numpy())

        _, _, height, widht = feature_map.shape

        # spatial masked feature map generation
        spatial_masked_feature_map = self._mosaic_feature(
            feature_map, is_gaussian=True
        )

        # reciprocal logit calculation
        reciprocal_predictions = self.head_net(spatial_masked_feature_map)
        reciprocal_logits = self.softmax(reciprocal_predictions)

        # Class activation map (CAM) generation
        cam = self._weight_accum(reciprocal_logits, class_index, height, widht)

        # Normalization
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, class_index
