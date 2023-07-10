from collections import OrderedDict
from typing import Union, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


_state_dict_url = {
    "repvgg_a0": "https://github.com/wonbeomjang/alembic/releases/download/parameter/RepVGG-A0-train.pth",
    "repvgg_a1": "https://github.com/wonbeomjang/alembic/releases/download/parameter/RepVGG-A1-train.pth",
    "repvgg_a2": "https://github.com/wonbeomjang/alembic/releases/download/parameter/RepVGG-A2-train.pth",
    "repvgg_b0": "https://github.com/wonbeomjang/alembic/releases/download/parameter/RepVGG-B0-train.pth",
    "repvgg_b1": "https://github.com/wonbeomjang/alembic/releases/download/parameter/RepVGG-B1-train.pth",
    "repvgg_b2": "https://github.com/wonbeomjang/alembic/releases/download/parameter/RepVGG-B2-train.pth",
    "repvgg_b3": "https://github.com/wonbeomjang/alembic/releases/download/parameter/RepVGG-B3-train.pth",
}


cfg = {
    "repvgg_a0": {
        "num_block": [2, 4, 14, 1],
        "width_multiplier": [0.75, 0.75, 0.75, 2.5],
    },
    "repvgg_a1": {
        "num_block": [2, 4, 14, 1],
        "width_multiplier": [1, 1, 1, 2.5],
    },
    "repvgg_a2": {
        "num_block": [2, 4, 14, 1],
        "width_multiplier": [1.5, 1.5, 1.5, 2.75],
    },
    "repvgg_b0": {
        "num_block": [4, 6, 16, 1],
        "width_multiplier": [1, 1, 1, 2.5],
    },
    "repvgg_b1": {
        "num_block": [4, 6, 16, 1],
        "width_multiplier": [2, 2, 2, 4],
    },
    "repvgg_b2": {
        "num_block": [4, 6, 16, 1],
        "width_multiplier": [2.5, 2.5, 2.5, 5],
    },
    "repvgg_b3": {
        "num_block": [4, 6, 16, 1],
        "width_multiplier": [3, 3, 3, 5],
    },
}


class ReparameterizableModel(nn.Module):
    def reparameterize(self):
        pass


class SEBlock:
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(
            in_channels=input_channels,
            out_channels=internal_neurons,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.up = nn.Conv2d(
            in_channels=internal_neurons,
            out_channels=input_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


def conv_bn(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    padding_mode: str = "zeros",
):
    block = nn.Sequential()
    block.add_module(
        "conv",
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            False,
            padding_mode,
        ),
    )
    block.add_module("bn", nn.BatchNorm2d(out_channels))

    return block


def _fuse_conv_bn(block: Union[nn.Sequential, nn.BatchNorm2d, None], groups: int = 1):
    if block is None:
        return 0, 0

    if isinstance(block, nn.Sequential):
        assert len(block) == 2
        bn = block.bn
        conv = block.conv
        conv_weight = conv.weight

    elif isinstance(block, nn.BatchNorm2d):
        bn = block
        input_dim = bn.num_features // groups
        kernel_value = np.zeros((bn.num_features, input_dim, 3, 3), dtype=np.float32)
        for i in range(bn.num_features):
            kernel_value[i, i % input_dim, 1, 1] = 1
        conv_weight = torch.from_numpy(kernel_value).to(bn.weight.device)

    else:
        raise AttributeError(
            f"block must be one of [nn.Sequential, nn.BatchNorm2d], but get {type(block)}"
        )

    bn_weight = bn.weight
    bn_std = (bn.running_var + bn.eps).sqrt()
    bn_bias = bn.bias

    return (bn_weight / bn_std).reshape(
        -1, 1, 1, 1
    ) * conv_weight, bn_bias - bn.running_mean * bn.weight / bn_std


def _pad_1x1(conv_weight, bias):
    return torch.nn.functional.pad(conv_weight, (1, 1, 1, 1)), bias


class RepVGGBlock(ReparameterizableModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        deploy: bool = False,
        use_se: bool = False,
    ):
        super().__init__()

        self.rbr_reparam = (
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                False,
                padding_mode,
            )
            if deploy
            else None
        )
        self.se = SEBlock(out_channels, out_channels // 16) if use_se else nn.Identity()
        self.nonlinearity = nn.ReLU()
        self.rbr_identity = (
            nn.BatchNorm2d(num_features=in_channels)
            if out_channels == in_channels and stride == 1
            else None
        )
        self.rbr_dense = conv_bn(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            padding_mode,
        )
        self.rbr_1x1 = conv_bn(
            in_channels,
            out_channels,
            1,
            stride,
            0,
            1,
            groups,
            padding_mode,
        )

    def forward(self, x):
        if self.rbr_reparam is not None:
            return self.nonlinearity(self.se(self.rbr_reparam(x)))

        if self.rbr_identity is not None:
            return self.nonlinearity(
                self.se(self.rbr_dense(x) + self.rbr_1x1(x) + self.rbr_identity(x))
            )
        return self.nonlinearity(self.se(self.rbr_dense(x) + self.rbr_1x1(x)))

    def reparameterize(self):
        weight_3x3, bias_3x3 = _fuse_conv_bn(self.rbr_dense)
        weight_1x1, bias_1x1 = _pad_1x1(*_fuse_conv_bn(self.rbr_1x1))
        weight_id, bias_id = _fuse_conv_bn(self.rbr_identity)

        state_dict = {
            "weight": weight_3x3 + weight_1x1 + weight_id,
            "bias": bias_3x3 + bias_1x1 + bias_id,
        }
        self.rbr_reparam = nn.Conv2d(
            self.rbr_dense.conv.in_channels,
            self.rbr_dense.conv.out_channels,
            self.rbr_dense.conv.kernel_size,
            self.rbr_dense.conv.stride,
            self.rbr_dense.conv.padding,
            self.rbr_dense.conv.dilation,
            self.rbr_dense.conv.groups,
            True,
            self.rbr_dense.conv.padding_mode,
        ).to(self.rbr_dense.conv.weight.device)

        self.rbr_reparam.load_state_dict(state_dict)
        delattr(self, "rbr_identity")
        delattr(self, "rbr_dense")
        delattr(self, "rbr_1x1")


def _make_stage(
    in_plane: int, out_plane: int, num_block: int, deploy: bool, use_se: bool
):
    strides = [2] + [1] * (num_block - 1)
    blocks = []
    for stride in strides:
        blocks += [
            RepVGGBlock(
                in_plane, out_plane, stride=stride, deploy=deploy, use_se=use_se
            )
        ]
        in_plane = out_plane

    return nn.Sequential(*blocks)


class RepVGG(nn.Module):
    def __init__(
        self,
        width_multiplier: List[float],
        num_block: List[int],
        deploy: bool = False,
        use_se: bool = False,
    ):
        super().__init__()

        in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(
            in_channels=3,
            out_channels=in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            deploy=deploy,
            use_se=use_se,
        )

        self.stage1 = _make_stage(
            in_planes, int(64 * width_multiplier[0]), num_block[0], deploy, use_se
        )
        self.stage2 = _make_stage(
            int(64 * width_multiplier[0]),
            int(128 * width_multiplier[1]),
            num_block[1],
            deploy,
            use_se,
        )
        self.stage3 = _make_stage(
            int(128 * width_multiplier[1]),
            int(256 * width_multiplier[2]),
            num_block[2],
            deploy,
            use_se,
        )
        self.stage4 = _make_stage(
            int(256 * width_multiplier[2]),
            int(512 * width_multiplier[3]),
            num_block[3],
            deploy,
            use_se,
        )

    def forward(self, x):
        result = OrderedDict()
        result["0"] = self.stage0(x)
        result["1"] = self.stage1(result["0"])
        result["2"] = self.stage2(result["1"])
        result["3"] = self.stage3(result["2"])
        result["4"] = self.stage4(result["3"])
        return result

    def reparameterize(self):
        for m in self.modules():
            if isinstance(m, ReparameterizableModel):
                m.reparameterize()

        return self


def rep_vgg(net_type: str, pretrained: bool = True):
    model = RepVGG(
        width_multiplier=cfg[net_type]["width_multiplier"],
        num_block=cfg[net_type]["num_block"],
    )

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_state_dict_url[net_type])
        model.load_state_dict(state_dict)
    return model
