from absl.testing import parameterized

from vision.configs import optimizer as optimizer_config
from vision.optimizer import get_optimizer


class Test(parameterized.TestCase):
    @parameterized.parameters(["adam"])
    def test_optimizer(self, optimizer_type):
        optimizer_cfg = optimizer_config.Optimizer(type=optimizer_type)
        get_optimizer(optimizer_cfg)
