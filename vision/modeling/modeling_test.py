from absl.testing import parameterized
import torch

from vision.configs import backbones
from vision.modeling import get_model


class Test(parameterized.TestCase):
    @parameterized.parameters(
        ("resnet18",),
        ("resnet34",),
        ("resnet50",),
        ("resnet101",),
        ("resnet152",),
        ("resnext50_32x4d",),
        ("resnext101_32x8d",),
        ("resnext101_64x4d",),
        ("wide_resnet50_2",),
        ("wide_resnet101_2",),
    )
    def test_alembic_resnet(self, model_id):
        backbone_cfg = backbones.Backbone(type="alembic_resnet")
        backbone_cfg.alembic_resnet.model_id = model_id

        model = get_model(backbone_cfg)

        result = model(torch.randn(1, 3, 256, 256))

        self.assertEqual(len(result.keys()), 5)
