import time
import logging

from absl.testing import parameterized
import torch

from vision.configs import backbones, classification, detection, unet
from vision.modeling.backbones import get_backbone as get_backbone_model
from vision.modeling import get_model

from vision.modeling.backbones import resnet
from vision.modeling.backbones import mobilenet
from vision.modeling.backbones import ghostnet
from vision.modeling.backbones import repvgg

console_logger = logging.getLogger("lightning.pytorch.core")
handler = logging.FileHandler("core.log")
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

console_logger.addHandler(handler)


def base_test(backbone_cfg, backbone_type, model_id):
    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        input_size = 256
        image_tensor = torch.randn(1, 3, input_size, input_size).to(device)

        # test backbone
        backbone = get_backbone_model(backbone_cfg).to(device)
        if hasattr(backbone, "reparameterize") and False:
            backbone.reparameterize()

        backbone(image_tensor)
        start_time = time.perf_counter()
        for i in range(100):
            backbone(image_tensor)
        inference_time = (time.perf_counter() - start_time) * 10
        console_logger.info(
            f"{backbone_type} {model_id} on backbone: {inference_time:.4f}ms"
        )

        # test classification
        classification_cfg = classification.ClassificationModel(num_classes=1000)
        classification_cfg.backbone = backbone_cfg

        model = get_model(classification_cfg).to(device)
        if hasattr(model.backbone, "reparameterize") and False:
            model.backbone.reparameterize()

        model(image_tensor)
        start_time = time.perf_counter()
        for i in range(100):
            model(image_tensor)
        inference_time = (time.perf_counter() - start_time) * 10
        console_logger.info(
            f"{backbone_type} {model_id} on classification_model: {inference_time:.4f}ms"
        )

        # test unet
        segmentation_cfg = unet.Segmentation()
        segmentation_cfg.type = "unet"

        segmentation_cfg.unet.backbone = backbone_cfg
        model = get_model(segmentation_cfg).to(device)
        if hasattr(model.backbone, "reparameterize") and False:
            model.backbone.reparameterize()

        model(image_tensor)
        start_time = time.perf_counter()
        for i in range(100):
            model(image_tensor)
        inference_time = (time.perf_counter() - start_time) * 10
        console_logger.info(
            f"{backbone_type} {model_id} on unet: {inference_time:.4f}ms"
        )

        # test yolo
        detection_cfg = detection.DetectionModel(type="yolo", num_classes=80)
        detection_cfg.type = "yolo"
        detection_cfg.yolo.backbone = backbone_cfg

        model = get_model(detection_cfg).to(device)
        if hasattr(model.backbone, "reparameterize") and False:
            model.backbone.reparameterize()

        image_tensor = (
            torch.randn(3, input_size, input_size).to(device),
            torch.randn(3, input_size, input_size).to(device),
        )

        model(image_tensor)
        start_time = time.perf_counter()
        for i in range(100):
            model(image_tensor)
        inference_time = (time.perf_counter() - start_time) * 10
        console_logger.info(
            f"{backbone_type} {model_id} on yolo: {inference_time:.4f}ms"
        )


class Test(parameterized.TestCase):
    @parameterized.parameters(*resnet.support_model)
    def test_alembic_resnet(self, model_id):
        backbone_cfg = backbones.Backbone(type="alembic_resnet")
        backbone_cfg.alembic_resnet.model_id = model_id

        base_test(backbone_cfg, "resnet", model_id)

    @parameterized.parameters(*mobilenet.support_model)
    def test_alembic_mobilenet(self, model_id):
        backbone_cfg = backbones.Backbone(type="alembic_mobilenet")
        backbone_cfg.alembic_mobilenet.model_id = model_id

        base_test(backbone_cfg, "mobilenet", model_id)

    @parameterized.parameters(*ghostnet.support_model)
    def test_alembic_ghostnet(self, model_id):
        backbone_cfg = backbones.Backbone(type="alembic_ghostnet")
        backbone_cfg.alembic_ghostnet.model_id = model_id

        base_test(backbone_cfg, "ghostnet", model_id)

    @parameterized.parameters(*repvgg.support_model)
    def test_alembic_repvgg(self, model_id):
        backbone_cfg = backbones.Backbone(type="alembic_repvgg")
        backbone_cfg.alembic_repvgg.model_id = model_id

        base_test(backbone_cfg, "repvgg", model_id)
