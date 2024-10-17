import torch
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer

# DINO 모델을 META_ARCH_REGISTRY에 등록하는 함수를 정의
def register_dino_model():
    from detectron2.modeling import META_ARCH_REGISTRY
    from models.dino import build_dino

    # DINOModel을 META_ARCH_REGISTRY에 등록
    @META_ARCH_REGISTRY.register()
    class DINOModel(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.model = build_dino()

        def forward(self, batched_inputs):
            return self.model(batched_inputs)

# DINO 모델을 등록하는 함수를 호출
register_dino_model()

