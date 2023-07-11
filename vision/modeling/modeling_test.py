from typing import Dict

from absl.testing import parameterized
import torch
from torch import Tensor

from vision.configs import backbones, classification, detection, segmentation
from vision.modeling.backbones import get_backbone as get_backbone_model
from vision.modeling import get_model

from vision.modeling.backbones import resnet
from vision.modeling.backbones import mobilenet
from vision.modeling.backbones import ghostnet
from vision.modeling.backbones import repvgg


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

    @parameterized.parameters(*repvgg.support_model)
    def test_alembic_repvgg(self, model_id):
        backbone_cfg = backbones.Backbone(type="alembic_repvgg")
        backbone_cfg.alembic_repvgg.model_id = model_id

        self.base_test(backbone_cfg)

    def base_test(self, backbone_cfg):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        image_size = 256
        num_classes = 100
        image_tensor: Tensor = torch.randn(1, 3, image_size, image_size).to(device)
        detection_input_data = (
            torch.randn(3, image_size, image_size).to(device),
            torch.randn(3, image_size, image_size).to(device),
        )

        # test backbone
        backbone = get_backbone_model(backbone_cfg).to(device)
        result: Dict[str, Tensor] = backbone(image_tensor)
        self.assertEqual(5, len(result.keys()))
        for i in range(5):
            self.assertEqual(image_size // (2 ** (i + 1)), result[str(i)].size(2))

        # test classification
        classification_cfg = classification.ClassificationModel(
            backbone=backbone_cfg, num_classes=num_classes
        )
        model = get_model(classification_cfg).to(device)

        result: Tensor = model(image_tensor)
        self.assertEqual(result.shape, torch.Size([1, num_classes]))

        # test yolo
        detection_cfg = detection.DetectionModel(type="yolo", num_classes=num_classes)
        detection_cfg.type = "yolo"
        detection_cfg.yolo.backbone = backbone_cfg

        model = get_model(detection_cfg).to(device)
        result: Dict[str, Tensor] = model(detection_input_data)

        self.assertEqual(result["boxes"].size(2), 4)
        self.assertEqual(result["labels"].size(2), num_classes + 1)

        # test unet
        segmentation_cfg = segmentation.Segmentation(
            type="unet",
            unet=segmentation.Unet(backbone=backbone_cfg, num_classes=num_classes),
        )
        model = get_model(segmentation_cfg).to(device)

        result: Tensor = model(image_tensor)
        self.assertEqual(result.size(1), segmentation_cfg.unet.num_classes)
        self.assertEqual(result.size(2), image_size)
