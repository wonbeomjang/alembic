from typing import Dict

from absl.testing import parameterized
import torch
from torch import Tensor

from vision.configs import backbones
from vision.configs import classification
from vision.configs import detection
from vision.modeling.backbones import get_backbone as get_backbone_model
from vision.modeling import get_model

from vision.modeling.backbones import resnet
from vision.modeling.backbones import mobilenet
from vision.modeling.backbones import ghostnet


class Test(parameterized.TestCase):
    @parameterized.parameters(*resnet.support_model)
    def test_alembic_resnet(self, model_id):
        backbone_cfg = backbones.Backbone(type="alembic_resnet")
        backbone_cfg.alembic_resnet.model_id = model_id

        self.base_test(backbone_cfg)

    @parameterized.parameters(*mobilenet.support_model)
    def test_alembic_mobilenet(self, model_id):
        backbone_cfg = backbones.Backbone(type="alembic_mobilenet")
        backbone_cfg.alembic_mobilenet.model_id = model_id

        self.base_test(backbone_cfg)

    @parameterized.parameters(*ghostnet.support_model)
    def test_alembic_ghostnet(self, model_id):
        backbone_cfg = backbones.Backbone(type="alembic_ghostnet")
        backbone_cfg.alembic_ghostnet.model_id = model_id

        self.base_test(backbone_cfg)

    def base_test(self, backbone_cfg):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        input_size = 256
        image_tensor = torch.randn(1, 3, input_size, input_size).to(device)

        # test backbone
        backbone = get_backbone_model(backbone_cfg).to(device)
        result: Dict[str, Tensor] = backbone(image_tensor)
        self.assertEqual(5, len(result.keys()))
        for i in range(5):
            self.assertEqual(input_size // (2 ** (i + 1)), result[str(i)].size(2))

        # test classification
        classification_cfg = classification.ClassificationModel()
        classification_cfg.backbone = backbone_cfg

        classification_model = get_model(classification_cfg).to(device)
        result: Tensor = classification_model(torch.randn(1, 3, 256, 256))
        self.assertEqual(result.shape, torch.Size([1, 1000]))

        # test yolo
        yolo_cfg = detection.YOLO()
        yolo_cfg.backbone = backbone_cfg

        model = get_model(yolo_cfg).to(device)
        result: Dict[str, Dict[str, Tensor]] = model(torch.randn(1, 3, 256, 256))
        for pyramid_level in range(
            yolo_cfg.head.yolo._min_level, yolo_cfg.head.yolo._max_level + 1
        ):
            self.assertEqual(result[str(pyramid_level)]["boxes"].size(2), 4)
            self.assertEqual(result[str(pyramid_level)]["background"].size(2), 1)
            self.assertEqual(
                result[str(pyramid_level)]["classes"].size(2),
                yolo_cfg.head._num_classes,
            )
