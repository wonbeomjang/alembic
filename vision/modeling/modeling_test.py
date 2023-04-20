from absl.testing import parameterized
import torch

from vision.configs import backbones
from vision.configs import classification
from vision.modeling.backbones import get_model as get_backbone_model
from vision.modeling import get_model

from vision.modeling.backbones import resnet


class Test(parameterized.TestCase):
    @parameterized.parameters(*resnet.support_model)
    def test_alembic_resnet(self, model_id):
        backbone_cfg = backbones.Backbone(type="alembic_resnet")
        backbone_cfg.alembic_resnet.model_id = model_id

        model = get_backbone_model(backbone_cfg)

        result = model(torch.randn(1, 3, 256, 256))

        self.assertEqual(len(result.keys()), 5)

    @parameterized.parameters(*resnet.support_model)
    def test_classification_model(self, backbone_id):
        classification_cfg = classification.ClassificationModel()
        classification_cfg.backbone.alembic_resnet.type = backbone_id

        model = get_model(classification_cfg)
        result = model(torch.randn(1, 3, 256, 256))

        self.assertEqual(result.shape, torch.Size([1, 1000]))
